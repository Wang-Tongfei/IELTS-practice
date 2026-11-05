#!/usr/bin/env python3
"""Offline accuracy and stability report for IELTS writing scores.

Without environment variables this never touches the network: it validates
the manifest and runs, then reports. With an OpenAI-compatible endpoint
configured, `--record` also scores the manifest's essays with a real model,
using the application's real scoring request and its real post-call rules:

    WRITING_EVAL_BASE_URL   e.g. https://api.openai.com/v1  (".../chat/completions" is appended)
    WRITING_EVAL_API_KEY    bearer token
    WRITING_EVAL_MODEL      model name

The request (built-in rubric + output contract + measured facts) and the
replay of raw replies (half-band snapping, ceilings, the second/third sample
and the median) both come from the Rust crate through the test hook
`writing_eval_hook_when_asked`, so this script needs cargo for `--record`
and `--rescore`. Raw replies are saved, so `--rescore` can re-apply changed
post-call rules to an old recording without paying for new calls.

Inputs:

  * a sample manifest: essay reference, task type, optional task prompt and
    the known criterion bands for each essay (see README.md for the schema);
  * one or more JSONL files of recorded or fabricated runs: the final scores
    a scoring configuration produced, optionally with the individual scoring
    samples and raw replies behind each score.

Output: per scoring configuration, the error against the known bands for
the four criteria and the overall band, the share within half a band, the
shape of the predicted distribution (including how much lands on 6-6.5),
and how consistent repeated evaluations of the same essay are.

Every successful evaluation counts once, so run every essay the same number
of times when comparing configurations.

    # bundled synthetic fixtures (their bands do not represent real levels)
    python developer/tests/tools/writing-eval/eval_writing.py

    # record a real model on a private set, then report
    python developer/tests/tools/writing-eval/eval_writing.py \\
        --manifest developer/tests/tools/writing-eval/private/manifest.json \\
        --record developer/tests/tools/writing-eval/private/runs-modelA.jsonl --repeats 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ENV_KEYS = ("WRITING_EVAL_BASE_URL", "WRITING_EVAL_API_KEY", "WRITING_EVAL_MODEL")
DEFAULT_MANIFEST = HERE / "fixtures" / "synthetic_manifest.json"
DEFAULT_RUNS = HERE / "fixtures" / "synthetic_runs.jsonl"

SCHEMA_VERSION = 1
CRITERIA = ("taskResponse", "coherence", "lexical", "grammar")
FIELDS = CRITERIA + ("overall",)
TASK_TYPES = ("task1", "task2")
BANDS = tuple(index / 2 for index in range(19))  # 0.0, 0.5 ... 9.0
# Reference overall buckets used to expose regression towards the middle.
BUCKETS = (
    ("<5.0", 0.0, 4.5),
    ("5.0-5.5", 5.0, 5.5),
    ("6.0-6.5", 6.0, 6.5),
    ("7.0-7.5", 7.0, 7.5),
    (">=8.0", 8.0, 9.0),
)
MIDDLE_LOW, MIDDLE_HIGH = 6.0, 6.5


class EvalInputError(ValueError):
    """Raised with every problem found in the inputs, one per line."""

    def __init__(self, issues: list[str]):
        super().__init__("\n".join(issues))
        self.issues = issues


@dataclass(frozen=True)
class Sample:
    id: str
    task_type: str
    essay_ref: str
    reference: dict[str, float]
    source: str
    task_prompt: str | None = None


@dataclass
class Run:
    sample_id: str
    config: str
    repeat: int
    final: dict[str, float] | None
    samples: list[dict[str, float]] = field(default_factory=list)
    error: str | None = None
    recorded_overall: float | None = None


# --------------------------------------------------------------------------
# Band arithmetic
# --------------------------------------------------------------------------


def is_band(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 9.0
        and float(value) * 2 == round(float(value) * 2)
    )


def derive_overall(bands: dict[str, float]) -> float:
    """Mean of the four criteria on the half-band scale; .25 and .75 go up.

    Mirrors the application rule (writing_evaluation.rs `score_from_bands`)
    so recorded scores and references are compared on the same footing.
    """
    mean = sum(bands[name] for name in CRITERIA) / 4
    return math.floor(mean * 2 + 0.5 + 1e-9) / 2


def with_overall(bands: dict[str, float]) -> dict[str, float]:
    return {**{name: float(bands[name]) for name in CRITERIA}, "overall": derive_overall(bands)}


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------


def _bands_issues(where: str, bands: Any) -> list[str]:
    if not isinstance(bands, dict):
        return [f"{where}: expected an object with {', '.join(CRITERIA)}"]
    issues = []
    for name in CRITERIA:
        if name not in bands:
            issues.append(f"{where}: missing {name}")
        elif not is_band(bands[name]):
            issues.append(f"{where}.{name}: {bands[name]!r} is not a band from 0 to 9 in steps of 0.5")
    return issues


def parse_manifest(raw: Any) -> tuple[dict[str, Sample], dict[str, Any]]:
    issues: list[str] = []
    if not isinstance(raw, dict) or not isinstance(raw.get("samples"), list):
        raise EvalInputError(["manifest: expected an object with a samples list"])
    if raw.get("schemaVersion") != SCHEMA_VERSION:
        issues.append(f"manifest: schemaVersion must be {SCHEMA_VERSION}")
    samples: dict[str, Sample] = {}
    for index, item in enumerate(raw["samples"]):
        where = f"samples[{index}]"
        if not isinstance(item, dict):
            issues.append(f"{where}: expected an object")
            continue
        sample_id = item.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            issues.append(f"{where}: id is required")
            continue
        where = f"sample {sample_id}"
        if sample_id in samples:
            issues.append(f"{where}: duplicate id")
        if item.get("taskType") not in TASK_TYPES:
            issues.append(f"{where}: taskType must be one of {', '.join(TASK_TYPES)}")
        if not isinstance(item.get("essayRef"), str) or not item["essayRef"]:
            issues.append(f"{where}: essayRef is required")
        if not isinstance(item.get("source"), str) or not item["source"]:
            issues.append(f"{where}: source is required (who wrote and who rated the essay)")
        if item.get("taskPrompt") is not None and not isinstance(item["taskPrompt"], str):
            issues.append(f"{where}: taskPrompt must be a string")
        reference_issues = _bands_issues(f"{where}.reference", item.get("reference"))
        issues.extend(reference_issues)
        if reference_issues:
            continue
        reference = with_overall(item["reference"])
        stated = item.get("referenceOverall")
        if stated is not None and stated != reference["overall"]:
            issues.append(
                f"{where}: referenceOverall {stated} disagrees with the criteria, which give {reference['overall']}"
            )
        samples[sample_id] = Sample(
            id=sample_id,
            task_type=item["taskType"],
            essay_ref=item["essayRef"],
            reference=reference,
            source=str(item.get("source", "")),
            task_prompt=item.get("taskPrompt"),
        )
    if issues:
        raise EvalInputError(issues)
    meta = {key: raw.get(key) for key in ("dataNotice", "synthetic", "description") if key in raw}
    return samples, meta


def parse_runs(lines: Iterable[tuple[str, str]], samples: dict[str, Sample]) -> list[Run]:
    """`lines` yields (location, text) pairs; blank lines and notice lines are skipped."""
    issues: list[str] = []
    runs: list[Run] = []
    seen: set[tuple[str, str, int]] = set()
    for where, text in lines:
        if not text.strip():
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError as error:
            issues.append(f"{where}: invalid JSON ({error.msg})")
            continue
        if not isinstance(item, dict):
            issues.append(f"{where}: expected an object")
            continue
        if "_notice" in item:
            continue
        sample_id, config, repeat = item.get("sampleId"), item.get("config"), item.get("repeat", 0)
        if sample_id not in samples:
            issues.append(f"{where}: sampleId {sample_id!r} is not in the manifest")
            continue
        if not isinstance(config, str) or not config:
            issues.append(f"{where}: config is required")
            continue
        if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 0:
            issues.append(f"{where}: repeat must be a non-negative integer")
            continue
        key = (sample_id, config, repeat)
        if key in seen:
            issues.append(f"{where}: duplicate run {sample_id}/{config}/repeat {repeat}")
            continue
        seen.add(key)
        error = item.get("error")
        final = item.get("final")
        if error is not None:
            if final is not None:
                issues.append(f"{where}: a failed run must not carry a final score")
                continue
            runs.append(Run(sample_id, config, repeat, None, error=str(error)))
            continue
        final_issues = _bands_issues(f"{where}.final", final)
        sample_scores = item.get("samples") or []
        if not isinstance(sample_scores, list):
            final_issues.append(f"{where}.samples: expected a list")
            sample_scores = []
        for index, score in enumerate(sample_scores):
            final_issues.extend(_bands_issues(f"{where}.samples[{index}]", score))
        recorded = final.get("overall") if isinstance(final, dict) else None
        if recorded is not None and not is_band(recorded):
            final_issues.append(f"{where}.final.overall: {recorded!r} is not a band")
        if final_issues:
            issues.extend(final_issues)
            continue
        runs.append(
            Run(
                sample_id,
                config,
                repeat,
                with_overall(final),
                samples=[with_overall(score) for score in sample_scores],
                recorded_overall=recorded,
            )
        )
    if issues:
        raise EvalInputError(issues)
    return runs


def load_manifest(path: Path) -> tuple[dict[str, Sample], dict[str, Any]]:
    return parse_manifest(json.loads(path.read_text(encoding="utf-8")))


def load_runs(paths: list[Path], samples: dict[str, Sample]) -> list[Run]:
    def lines() -> Iterable[tuple[str, str]]:
        for path in paths:
            for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                yield f"{path.name}:{number}", text

    return parse_runs(lines(), samples)


def missing_essays(samples: dict[str, Sample], base: Path) -> list[str]:
    return [sample.id for sample in samples.values() if not (base / sample.essay_ref).is_file()]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def quadratic_weighted_kappa(reference: list[float], predicted: list[float]) -> float | None:
    """Cohen's kappa with quadratic weights over the 19 half-band categories."""
    if not reference:
        return None
    size = len(BANDS)
    observed = [[0] * size for _ in range(size)]
    for ref, pred in zip(reference, predicted):
        observed[int(ref * 2)][int(pred * 2)] += 1
    total = len(reference)
    ref_hist = [sum(row) for row in observed]
    pred_hist = [sum(observed[i][j] for i in range(size)) for j in range(size)]
    numerator = denominator = 0.0
    for i in range(size):
        for j in range(size):
            weight = (i - j) ** 2 / (size - 1) ** 2
            numerator += weight * observed[i][j]
            denominator += weight * ref_hist[i] * pred_hist[j] / total
    if denominator == 0:
        return None
    return 1 - numerator / denominator


def accuracy(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    """`pairs` holds (reference, predicted) for one field."""
    if not pairs:
        return {"n": 0}
    refs = [ref for ref, _ in pairs]
    preds = [pred for _, pred in pairs]
    errors = [pred - ref for ref, pred in pairs]
    return {
        "n": len(pairs),
        "mae": _round(_mean([abs(error) for error in errors])),
        "bias": _round(_mean(errors)),
        "rmse": _round(math.sqrt(_mean([error**2 for error in errors]))),
        "exact": _round(_mean([float(error == 0) for error in errors])),
        "withinHalfBand": _round(_mean([float(abs(error) <= 0.5) for error in errors])),
        "withinOneBand": _round(_mean([float(abs(error) <= 1.0) for error in errors])),
        "qwk": _round(quadratic_weighted_kappa(refs, preds)),
        "pearson": _round(pearson(refs, preds)),
    }


def _histogram(values: list[float]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in sorted(values):
        key = f"{value:.1f}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _middle_share(values: list[float]) -> float | None:
    return _mean([float(MIDDLE_LOW <= value <= MIDDLE_HIGH) for value in values])


def distribution(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    """Shape of the overall band: is everything squeezed onto 6-6.5?"""
    refs = [ref for ref, _ in pairs]
    preds = [pred for _, pred in pairs]
    ref_std, pred_std = _std(refs), _std(preds)
    buckets = []
    for label, low, high in BUCKETS:
        bucket = [(ref, pred) for ref, pred in pairs if low <= ref <= high]
        buckets.append(
            {
                "referenceBand": label,
                "n": len(bucket),
                "mae": _round(_mean([abs(pred - ref) for ref, pred in bucket])),
                "bias": _round(_mean([pred - ref for ref, pred in bucket])),
            }
        )
    return {
        "predictedHistogram": _histogram(preds),
        "referenceHistogram": _histogram(refs),
        "predictedShare6to6_5": _round(_middle_share(preds)),
        "referenceShare6to6_5": _round(_middle_share(refs)),
        "predictedStd": _round(pred_std),
        "referenceStd": _round(ref_std),
        # Below 1: predictions are more bunched than the references.
        "spreadRatio": _round(pred_std / ref_std) if ref_std and pred_std is not None else None,
        "byReferenceBand": buckets,
    }


def _spread(scores: list[dict[str, float]], name: str) -> float:
    values = [score[name] for score in scores]
    return max(values) - min(values)


def repeat_consistency(groups: list[list[dict[str, float]]]) -> dict[str, Any]:
    """`groups` holds, per essay, the final scores of its repeated evaluations."""
    groups = [group for group in groups if len(group) >= 2]
    result: dict[str, Any] = {"essaysWithRepeats": len(groups)}
    if not groups:
        return result
    result["overallIdentical"] = _round(_mean([float(_spread(group, "overall") == 0) for group in groups]))
    per_field = {}
    for name in FIELDS:
        pair_diffs = [abs(a[name] - b[name]) for group in groups for a, b in combinations(group, 2)]
        spreads = [_spread(group, name) for group in groups]
        per_field[name] = {
            "meanPairDifference": _round(_mean(pair_diffs)),
            "meanSpread": _round(_mean(spreads)),
            "maxSpread": max(spreads),
            "spreadWithinHalfBand": _round(_mean([float(spread <= 0.5) for spread in spreads])),
        }
    result["byField"] = per_field
    return result


def sampling_profile(runs: list[Run], disagreement: float = 1.0) -> dict[str, Any]:
    """What the per-evaluation scoring samples say about cost and agreement.

    `disagreement` mirrors DISAGREEMENT_BANDS in writing_evaluation.rs.
    """
    with_samples = [run for run in runs if run.samples]
    result: dict[str, Any] = {"runsWithSamples": len(with_samples)}
    if not with_samples:
        return result
    first_pairs = [run.samples[:2] for run in with_samples if len(run.samples) >= 2]
    first_spreads = [max(_spread(pair, name) for name in CRITERIA) for pair in first_pairs]
    result.update(
        {
            "meanScoringCalls": _round(_mean([float(len(run.samples)) for run in with_samples])),
            "thirdSampleRate": _round(_mean([float(len(run.samples) >= 3) for run in with_samples])),
            "firstPairDisagreementRate": _round(
                _mean([float(spread >= disagreement) for spread in first_spreads])
            ),
            "firstPairMaxCriterionSpread": _histogram(first_spreads),
            # Agreeing pairs that differ by half a band are averaged and the
            # .25/.75 is snapped upward, so this share is pushed up half a band.
            "firstPairHalfBandSplitRate": _round(
                _mean([float(any(_spread(pair, name) == 0.5 for name in CRITERIA)) for pair in first_pairs])
            ),
        }
    )
    return result


def config_report(config: str, samples: dict[str, Sample], runs: list[Run]) -> dict[str, Any]:
    ok = [run for run in runs if run.final is not None]
    failed = [run for run in runs if run.final is None]
    pairs = {name: [(samples[run.sample_id].reference[name], run.final[name]) for run in ok] for name in FIELDS}
    by_task = {}
    for task_type in TASK_TYPES:
        task_runs = [run for run in ok if samples[run.sample_id].task_type == task_type]
        if task_runs:
            by_task[task_type] = {
                "n": len(task_runs),
                "overall": accuracy(
                    [(samples[run.sample_id].reference["overall"], run.final["overall"]) for run in task_runs]
                ),
            }
    groups: dict[str, list[dict[str, float]]] = {}
    for run in ok:
        groups.setdefault(run.sample_id, []).append(run.final)
    mismatches = [
        f"{run.sample_id}/repeat {run.repeat}: recorded {run.recorded_overall}, criteria give {run.final['overall']}"
        for run in ok
        if run.recorded_overall is not None and run.recorded_overall != run.final["overall"]
    ]
    return {
        "config": config,
        "coverage": {
            "manifestEssays": len(samples),
            "essaysScored": len(groups),
            "essaysMissing": sorted(set(samples) - set(groups)),
            "evaluations": len(runs),
            "failedEvaluations": len(failed),
            "failureRate": _round(len(failed) / len(runs)) if runs else None,
        },
        "accuracy": {name: accuracy(pairs[name]) for name in FIELDS},
        "accuracyByTask": by_task,
        "distribution": distribution(pairs["overall"]),
        "repeatConsistency": repeat_consistency(list(groups.values())),
        "sampling": sampling_profile(ok),
        "overallMismatches": mismatches,
    }


def build_report(
    samples: dict[str, Sample],
    runs: list[Run],
    meta: dict[str, Any],
    configs: list[str] | None = None,
) -> dict[str, Any]:
    names = sorted({run.config for run in runs})
    if configs:
        unknown = sorted(set(configs) - set(names))
        if unknown:
            raise EvalInputError([f"config {name!r} has no runs" for name in unknown])
        names = [name for name in names if name in configs]
    return {
        "schemaVersion": SCHEMA_VERSION,
        "dataNotice": meta.get("dataNotice"),
        "synthetic": bool(meta.get("synthetic")),
        "references": {
            "essays": len(samples),
            "byTask": {task: sum(s.task_type == task for s in samples.values()) for task in TASK_TYPES},
            "overallHistogram": _histogram([s.reference["overall"] for s in samples.values()]),
        },
        "configs": [config_report(name, samples, [run for run in runs if run.config == name]) for name in names],
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") if value != int(value) else f"{value:.1f}"
    return str(value)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Writing scoring eval report", ""]
    if report["synthetic"]:
        lines += ["> SYNTHETIC FIXTURES: these numbers only exercise the harness and say nothing about real scoring quality.", ""]
    if report.get("dataNotice"):
        lines += [f"Data notice: {report['dataNotice']}", ""]
    refs = report["references"]
    lines += [
        f"Reference essays: {refs['essays']} (task1 {refs['byTask']['task1']}, task2 {refs['byTask']['task2']}); "
        f"overall histogram {refs['overallHistogram']}",
        "",
        "## Overall band",
        "",
        "| config | n | failed | MAE | bias | within 0.5 | QWK | share 6-6.5 (pred/ref) | spread ratio | repeat identical |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for config in report["configs"]:
        overall = config["accuracy"]["overall"]
        dist = config["distribution"]
        rep = config["repeatConsistency"]
        lines.append(
            "| {name} | {n} | {failed} | {mae} | {bias} | {within} | {qwk} | {pred}/{ref} | {ratio} | {ident} |".format(
                name=config["config"],
                n=overall["n"],
                failed=config["coverage"]["failedEvaluations"],
                mae=_fmt(overall.get("mae")),
                bias=_fmt(overall.get("bias")),
                within=_pct(overall.get("withinHalfBand")),
                qwk=_fmt(overall.get("qwk")),
                pred=_pct(dist["predictedShare6to6_5"]),
                ref=_pct(dist["referenceShare6to6_5"]),
                ratio=_fmt(dist["spreadRatio"]),
                ident=_pct(rep.get("overallIdentical")),
            )
        )
    lines += ["", "## Criteria", "", "| config | field | MAE | bias | within 0.5 | QWK | repeat mean pair diff |", "|---|---|---|---|---|---|---|"]
    for config in report["configs"]:
        by_field = config["repeatConsistency"].get("byField", {})
        for name in FIELDS:
            acc = config["accuracy"][name]
            lines.append(
                f"| {config['config']} | {name} | {_fmt(acc.get('mae'))} | {_fmt(acc.get('bias'))} | "
                f"{_pct(acc.get('withinHalfBand'))} | {_fmt(acc.get('qwk'))} | "
                f"{_fmt(by_field.get(name, {}).get('meanPairDifference'))} |"
            )
    lines += ["", "## Regression to the middle (overall, by reference band)", "", "| config | reference band | n | MAE | bias |", "|---|---|---|---|---|"]
    for config in report["configs"]:
        for bucket in config["distribution"]["byReferenceBand"]:
            if bucket["n"]:
                lines.append(
                    f"| {config['config']} | {bucket['referenceBand']} | {bucket['n']} | {_fmt(bucket['mae'])} | {_fmt(bucket['bias'])} |"
                )
    sampled = [config for config in report["configs"] if config["sampling"]["runsWithSamples"]]
    if sampled:
        lines += ["", "## Scoring samples", "", "| config | runs | mean scoring calls | third-sample rate | first pair half-band split |", "|---|---|---|---|---|"]
        for config in sampled:
            sampling = config["sampling"]
            lines.append(
                f"| {config['config']} | {sampling['runsWithSamples']} | {_fmt(sampling['meanScoringCalls'])} | "
                f"{_pct(sampling['thirdSampleRate'])} | {_pct(sampling['firstPairHalfBandSplitRate'])} |"
            )
    for config in report["configs"]:
        if config["overallMismatches"]:
            lines += ["", f"Overall mismatches in {config['config']}:"] + [f"- {item}" for item in config["overallMismatches"]]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Recording a real model (only with WRITING_EVAL_* set and --record)
# --------------------------------------------------------------------------

# Mirrors src-tauri/src/ai/runtime.rs: OUTPUT_LIMIT_RULES / apply_output_controls
# (reasoning models take max_completion_tokens with a floor and no temperature)
# and CACHE_ROUTING_RULES (the cache key goes only where the endpoint documents
# a routing field). The coach evaluator sends temperature unconditionally,
# which those models reject.
REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")
REASONING_OUTPUT_FLOOR = 1024
CACHE_ROUTING = (("api.openai.com", "prompt_cache_key"), ("openrouter.ai", "session_id"))
RECORDING_NOTICE = (
    "Recorded with a real model. Replies may quote the essays: keep this file out of git "
    "(private/ is ignored) unless every essay is synthetic."
)

Hook = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
Ask = Callable[[dict[str, Any]], "str | None"]


def request_body(model: str, base_url: str, request: dict[str, Any]) -> dict[str, Any]:
    """The chat-completions body the application's gateway would send."""
    body: dict[str, Any] = {"model": model, "messages": request["messages"]}
    name = model.rsplit("/", 1)[-1].lower()
    if name.startswith(REASONING_PREFIXES):
        body["max_completion_tokens"] = max(int(request["maxTokens"]), REASONING_OUTPUT_FLOOR)
    else:
        body["max_tokens"] = int(request["maxTokens"])
        body["temperature"] = round(float(request["temperature"]), 4)
    host = base_url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0].lower()
    key = request.get("cacheKey")
    for suffix, field_name in CACHE_ROUTING:
        if key and (host == suffix or host.endswith("." + suffix)):
            body[field_name] = key
    return body


def http_ask(base_url: str, api_key: str, model: str) -> Ask:
    """One chat completion per call; a failed call returns None, like a dropped sample."""

    def ask(request: dict[str, Any]) -> str | None:
        data = json.dumps(request_body(model, base_url, request)).encode()
        http = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(http, timeout=180) as response:
                return json.load(response)["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, IndexError, TypeError, ValueError, TimeoutError) as error:
            print(f"request failed ({type(error).__name__}: {error})", file=sys.stderr)
            return None

    return ask


def cargo_hook(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Runs `writing_eval_hook_when_asked` in ielts-application (see its doc comment)."""
    environment = {name: value for name, value in os.environ.items() if name not in ENV_KEYS}
    with tempfile.TemporaryDirectory() as scratch:
        source, target = Path(scratch) / "in.json", Path(scratch) / "out.json"
        source.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        environment.update({"WRITING_EVAL_HOOK": str(source), "WRITING_EVAL_HOOK_OUT": str(target)})
        command = ["cargo", "test", "-q", "-p", "ielts-application", "--lib", "writing_eval_hook_when_asked"]
        done = subprocess.run(
            command, cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if done.returncode != 0 or not target.is_file():
            output = (done.stdout + done.stderr)[-800:]
            raise EvalInputError([f"the writing_eval_hook_when_asked cargo test failed:\n{output}"])
        return json.loads(target.read_text(encoding="utf-8"))


def load_essays(samples: dict[str, Sample], base: Path) -> dict[str, str]:
    missing = missing_essays(samples, base)
    if missing:
        raise EvalInputError([f"sample {sample_id}: essay file not found" for sample_id in missing])
    return {sample.id: (base / sample.essay_ref).read_text(encoding="utf-8") for sample in samples.values()}


def _hook_item(sample: Sample, essay: str, key: str, replies: list | None = None) -> dict[str, Any]:
    item = {"id": key, "taskType": sample.task_type, "prompt": sample.task_prompt, "essay": essay}
    if replies is not None:
        item["replies"] = replies
    return item


def prompt_fingerprint(requests: Iterable[dict[str, Any]]) -> str:
    """Hash of the system prompts actually sent, so a config names its rubric."""
    systems = sorted({request["messages"][0]["content"] for request in requests})
    return hashlib.sha256("\n\0\n".join(systems).encode()).hexdigest()[:12]


def _run_record(sample_id: str, repeat: int, replay: dict[str, Any], replies: list, meta: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {"sampleId": sample_id, **meta, "repeat": repeat}
    if "error" in replay:
        record["error"] = replay["error"]
    else:
        record.update(final=replay["final"], samples=replay["samples"], ceilings=replay["ceilings"])
        if replay.get("needsThird"):
            record["needsThird"] = True
    record["replies"] = replies
    return record


def replay_all(
    jobs: dict[str, tuple[Sample, int]],
    replies: dict[str, list],
    essays: dict[str, str],
    hook: Hook,
) -> dict[str, dict[str, Any]]:
    items = [_hook_item(sample, essays[sample.id], key, replies[key]) for key, (sample, _) in jobs.items()]
    return {item["id"]: item["replay"] for item in hook(items)} if items else {}


def record_runs(
    samples: dict[str, Sample],
    essays: dict[str, str],
    repeats: int,
    ask: Ask,
    hook: Hook,
    model: str,
    label: str | None = None,
) -> list[dict[str, Any]]:
    """Two scoring calls per evaluation, a third only where the application would make one."""
    requests = {
        item["id"]: item["request"]
        for item in hook([_hook_item(sample, essays[sample.id], sample.id) for sample in samples.values()])
    }
    fingerprint = prompt_fingerprint(requests.values())
    meta = {"config": label or f"{model}/{fingerprint}", "model": model, "promptFingerprint": fingerprint}
    jobs = {f"{sample.id}#{repeat}": (sample, repeat) for repeat in range(repeats) for sample in samples.values()}
    replies = {key: [ask(requests[sample.id]), ask(requests[sample.id])] for key, (sample, _) in jobs.items()}
    replays = replay_all(jobs, replies, essays, hook)
    third = {key: jobs[key] for key, replay in replays.items() if replay.get("needsThird")}
    for key, (sample, _) in third.items():
        replies[key].append(ask(requests[sample.id]))
    replays.update(replay_all(third, replies, essays, hook))
    return [_run_record(sample.id, repeat, replays[key], replies[key], meta) for key, (sample, repeat) in jobs.items()]


def rescore_runs(
    samples: dict[str, Sample],
    essays: dict[str, str],
    recorded: list[dict[str, Any]],
    hook: Hook,
    label: str | None = None,
) -> list[dict[str, Any]]:
    """Re-applies the current post-call rules to recorded raw replies; no network."""
    issues = []
    jobs: dict[str, tuple[Sample, int]] = {}
    replies: dict[str, list] = {}
    metas: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(recorded):
        sample = samples.get(item.get("sampleId"))
        if sample is None or not isinstance(item.get("replies"), list):
            issues.append(f"recorded[{index}]: needs a manifest sampleId and the raw replies list")
            continue
        key = f"{index}"
        jobs[key] = (sample, int(item.get("repeat", 0)))
        replies[key] = item["replies"]
        metas[key] = {name: item[name] for name in ("config", "model", "promptFingerprint") if name in item}
        if label:
            metas[key]["config"] = label
    if issues:
        raise EvalInputError(issues)
    replays = replay_all(jobs, replies, essays, hook)
    return [_run_record(sample.id, repeat, replays[key], replies[key], metas[key]) for key, (sample, repeat) in jobs.items()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for text in path.read_text(encoding="utf-8").splitlines():
        if text.strip():
            row = json.loads(text)
            if isinstance(row, dict) and "_notice" not in row:
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"_notice": RECORDING_NOTICE}, ensure_ascii=False)]
    lines += [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row_lines(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(f"row {index}", json.dumps(row)) for index, row in enumerate(rows, 1)]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runs", type=Path, action="append", help="JSONL of recorded scores; repeatable")
    parser.add_argument("--config", action="append", help="only report these configs; repeatable")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--check-essays", action="store_true", help="fail when an essayRef is not a file next to the manifest")
    parser.add_argument("--validate-only", action="store_true", help="check the manifest and runs and stop, even if the endpoint variables are set")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", type=Path, metavar="OUT", help="score the manifest with the WRITING_EVAL_* model and write the runs here")
    mode.add_argument("--rescore", type=Path, metavar="RECORDED", help="re-apply the current rules to recorded raw replies (no network)")
    parser.add_argument("--out", type=Path, help="where --rescore writes its runs")
    parser.add_argument("--repeats", type=int, default=2, help="evaluations per essay for --record (default 2)")
    parser.add_argument("--label", help="config name for --record/--rescore (default: <model>/<prompt fingerprint>)")
    return parser


def _produce_runs(args: argparse.Namespace, samples: dict[str, Sample]) -> list[Run]:
    if not (args.record or args.rescore):
        return load_runs(args.runs or [DEFAULT_RUNS], samples)
    essays = load_essays(samples, args.manifest.parent)
    if args.record:
        if args.repeats < 1:
            raise EvalInputError(["--repeats must be at least 1"])
        if not all(os.environ.get(name) for name in ENV_KEYS):
            raise EvalInputError([f"--record needs {', '.join(ENV_KEYS)}"])
        base_url, api_key, model = (os.environ[name] for name in ENV_KEYS)
        calls = len(samples) * args.repeats
        print(f"recording {calls} evaluations ({2 * calls}+ scoring calls) with {model}", file=sys.stderr)
        rows = record_runs(samples, essays, args.repeats, http_ask(base_url, api_key, model), cargo_hook, model, args.label)
        out = args.record
    else:
        if not args.out:
            raise EvalInputError(["--rescore needs --out"])
        rows = rescore_runs(samples, essays, read_jsonl(args.rescore), cargo_hook, args.label)
        out = args.out
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} runs to {out}", file=sys.stderr)
    return parse_runs(_row_lines(rows), samples)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        samples, meta = load_manifest(args.manifest)
        if args.check_essays:
            load_essays(samples, args.manifest.parent)
        if args.validate_only:
            runs = load_runs(args.runs or [DEFAULT_RUNS], samples)
            print(f"writing eval inputs ok: {len(samples)} essays, {len(runs)} runs")
            return 0
        runs = _produce_runs(args, samples)
        report = build_report(samples, runs, meta, args.config)
    except EvalInputError as error:
        print("writing eval input error:\n" + str(error), file=sys.stderr)
        return 2
    if not (args.record or args.rescore):
        print(f"live recording skipped: set {', '.join(ENV_KEYS)} and pass --record to score a real model", file=sys.stderr)
    markdown = render_markdown(report)
    for path, text in ((args.json_out, json.dumps(report, ensure_ascii=False, indent=2) + "\n"), (args.markdown_out, markdown)):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    raise SystemExit(main())
