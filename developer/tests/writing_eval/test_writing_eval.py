#!/usr/bin/env python3
"""Unit tests for the offline writing scoring eval harness (stdlib only)."""
from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_writing_eval as ev  # noqa: E402


def bands(tr, cc, lr, gra):
    return {"taskResponse": tr, "coherence": cc, "lexical": lr, "grammar": gra}


def manifest(*samples):
    return {"schemaVersion": 1, "samples": list(samples)}


def sample(sample_id, reference, task_type="task2"):
    return {
        "id": sample_id,
        "taskType": task_type,
        "essayRef": f"essays/{sample_id}.txt",
        "source": "unit-test",
        "reference": reference,
    }


def run_lines(*items):
    return [(f"runs:{index}", json.dumps(item)) for index, item in enumerate(items, 1)]


class BandArithmeticTest(unittest.TestCase):
    def test_overall_rounds_quarter_marks_up_and_eighths_to_nearest(self):
        cases = [
            (bands(6, 6, 6, 6.5), 6.0),  # 6.125
            (bands(6, 6, 6, 7), 6.5),  # 6.25 goes up
            (bands(6.5, 6.5, 6.5, 6), 6.5),  # 6.375
            (bands(6.5, 6.5, 6.5, 7), 6.5),  # 6.625
            (bands(7, 7, 7, 6), 7.0),  # 6.75 goes up
            (bands(9, 9, 9, 9), 9.0),
            (bands(0, 0, 0, 0), 0.0),
        ]
        for criteria, expected in cases:
            with self.subTest(criteria=criteria):
                self.assertEqual(ev.derive_overall(criteria), expected)

    def test_only_half_bands_between_zero_and_nine_are_bands(self):
        for value in (0, 0.5, 6, 6.5, 9):
            self.assertTrue(ev.is_band(value), value)
        for value in (6.25, 9.5, -0.5, True, "6", None, math.nan):
            self.assertFalse(ev.is_band(value), value)


class MetricTest(unittest.TestCase):
    def test_accuracy_reports_error_bias_and_tolerance_rates(self):
        result = ev.accuracy([(6.0, 6.0), (6.0, 7.0), (5.0, 4.0)])
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mae"], 0.6667, places=4)
        self.assertEqual(result["bias"], 0.0)
        self.assertAlmostEqual(result["rmse"], math.sqrt(2 / 3), places=4)
        self.assertAlmostEqual(result["exact"], 0.3333, places=4)
        self.assertAlmostEqual(result["withinHalfBand"], 0.3333, places=4)
        self.assertEqual(result["withinOneBand"], 1.0)

    def test_empty_accuracy_has_no_numbers(self):
        self.assertEqual(ev.accuracy([]), {"n": 0})

    def test_quadratic_weighted_kappa_known_values(self):
        self.assertEqual(ev.quadratic_weighted_kappa([5.0, 6.0, 7.0], [5.0, 6.0, 7.0]), 1.0)
        # A constant prediction carries no information about the reference.
        self.assertAlmostEqual(ev.quadratic_weighted_kappa([5.0, 6.0, 7.0], [6.0, 6.0, 6.0]), 0.0)
        # Perfectly swapped neighbours are as bad as chance can be.
        self.assertAlmostEqual(ev.quadratic_weighted_kappa([0.0, 0.5], [0.5, 0.0]), -1.0)
        self.assertIsNone(ev.quadratic_weighted_kappa([6.0, 6.0], [6.0, 6.0]))
        self.assertIsNone(ev.quadratic_weighted_kappa([], []))

    def test_pearson_needs_variance_on_both_sides(self):
        self.assertAlmostEqual(ev.pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0)
        self.assertIsNone(ev.pearson([1.0, 2.0], [5.0, 5.0]))
        self.assertIsNone(ev.pearson([1.0], [1.0]))

    def test_distribution_exposes_bunching_on_six(self):
        result = ev.distribution([(5.0, 6.0), (7.0, 6.0), (6.0, 6.0), (8.0, 6.5)])
        self.assertEqual(result["predictedShare6to6_5"], 1.0)
        self.assertEqual(result["referenceShare6to6_5"], 0.25)
        self.assertEqual(result["predictedHistogram"], {"6.0": 3, "6.5": 1})
        self.assertLess(result["spreadRatio"], 0.2)
        by_band = {bucket["referenceBand"]: bucket for bucket in result["byReferenceBand"]}
        self.assertEqual(by_band["5.0-5.5"]["bias"], 1.0)
        self.assertEqual(by_band["7.0-7.5"]["bias"], -1.0)
        self.assertEqual(by_band[">=8.0"]["bias"], -1.5)
        self.assertEqual(by_band["<5.0"]["n"], 0)
        self.assertIsNone(by_band["<5.0"]["mae"])

    def test_spread_ratio_is_undefined_without_reference_spread(self):
        self.assertIsNone(ev.distribution([(6.0, 5.0), (6.0, 7.0)])["spreadRatio"])

    def test_repeat_consistency_compares_every_pair_of_repeats(self):
        a = ev.with_overall(bands(6, 6, 6, 6))
        b = ev.with_overall(bands(6, 6, 6.5, 7))  # overall 6.5
        c = ev.with_overall(bands(7, 7, 7, 7))
        result = ev.repeat_consistency([[a, b], [c, c], [a]])
        self.assertEqual(result["essaysWithRepeats"], 2)
        self.assertEqual(result["overallIdentical"], 0.5)
        overall = result["byField"]["overall"]
        self.assertEqual(overall["meanPairDifference"], 0.25)
        self.assertEqual(overall["maxSpread"], 0.5)
        self.assertEqual(overall["spreadWithinHalfBand"], 1.0)
        grammar = result["byField"]["grammar"]
        self.assertEqual(grammar["maxSpread"], 1.0)
        self.assertEqual(grammar["spreadWithinHalfBand"], 0.5)
        self.assertEqual(ev.repeat_consistency([[a]]), {"essaysWithRepeats": 0})

    def test_sampling_profile_counts_calls_and_half_band_splits(self):
        six = ev.with_overall(bands(6, 6, 6, 6))
        split = ev.with_overall(bands(6, 6, 6.5, 6))
        far = ev.with_overall(bands(7, 6, 6, 6))
        runs = [
            ev.Run("a", "c", 0, six, samples=[six, split]),
            ev.Run("b", "c", 0, six, samples=[six, far, six]),
            ev.Run("d", "c", 0, six),
        ]
        result = ev.sampling_profile(runs)
        self.assertEqual(result["runsWithSamples"], 2)
        self.assertEqual(result["meanScoringCalls"], 2.5)
        self.assertEqual(result["thirdSampleRate"], 0.5)
        self.assertEqual(result["firstPairDisagreementRate"], 0.5)
        self.assertEqual(result["firstPairHalfBandSplitRate"], 0.5)
        self.assertEqual(result["firstPairMaxCriterionSpread"], {"0.5": 1, "1.0": 1})


class InputValidationTest(unittest.TestCase):
    def test_manifest_problems_are_all_reported_together(self):
        raw = manifest(
            sample("a", bands(6, 6, 6, 6.25)),
            sample("b", bands(6, 6, 6, 6), task_type="task3"),
            sample("b", bands(6, 6, 6, 6)),
            {**sample("c", bands(6, 6, 6, 6)), "referenceOverall": 7.0},
            {"id": "d", "taskType": "task2", "essayRef": "x", "reference": bands(6, 6, 6, 6)},
        )
        with self.assertRaises(ev.EvalInputError) as caught:
            ev.parse_manifest(raw)
        text = str(caught.exception)
        self.assertIn("sample a.reference.grammar: 6.25", text)
        self.assertIn("sample b: taskType", text)
        self.assertIn("sample b: duplicate id", text)
        self.assertIn("referenceOverall 7.0 disagrees", text)
        self.assertIn("sample d: source is required", text)

    def test_manifest_derives_reference_overall(self):
        samples, _ = ev.parse_manifest(manifest(sample("a", bands(6, 6, 6, 7))))
        self.assertEqual(samples["a"].reference["overall"], 6.5)

    def test_run_problems_are_all_reported_together(self):
        samples, _ = ev.parse_manifest(manifest(sample("a", bands(6, 6, 6, 6))))
        good = {"sampleId": "a", "config": "x", "repeat": 0, "final": bands(6, 6, 6, 6)}
        lines = run_lines(
            good,
            good,
            {"sampleId": "zzz", "config": "x", "final": bands(6, 6, 6, 6)},
            {"sampleId": "a", "config": "x", "repeat": 1, "final": bands(6, 6, 6, 10)},
            {"sampleId": "a", "config": "x", "repeat": 2, "error": "boom", "final": bands(6, 6, 6, 6)},
            {"sampleId": "a", "config": "", "repeat": 3, "final": bands(6, 6, 6, 6)},
            {"sampleId": "a", "config": "x", "repeat": 4, "final": bands(6, 6, 6, 6), "samples": [bands(6, 6, 6, 6.3)]},
        ) + [("runs:8", "{not json")]
        with self.assertRaises(ev.EvalInputError) as caught:
            ev.parse_runs(lines, samples)
        issues = caught.exception.issues
        self.assertEqual(len(issues), 7, issues)
        self.assertIn("duplicate run a/x/repeat 0", issues[0])
        self.assertIn("'zzz' is not in the manifest", issues[1])
        self.assertIn("final.grammar: 10", issues[2])
        self.assertIn("failed run must not carry a final score", issues[3])
        self.assertIn("config is required", issues[4])
        self.assertIn("samples[0].grammar", issues[5])
        self.assertIn("invalid JSON", issues[6])

    def test_notice_and_blank_lines_are_skipped_and_failures_kept(self):
        samples, _ = ev.parse_manifest(manifest(sample("a", bands(6, 6, 6, 6))))
        lines = run_lines({"_notice": "fixture"}, {"sampleId": "a", "config": "x", "error": "timeout"})
        lines.append(("runs:3", "   "))
        runs = ev.parse_runs(lines, samples)
        self.assertEqual(len(runs), 1)
        self.assertIsNone(runs[0].final)
        self.assertEqual(runs[0].error, "timeout")


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.samples, self.meta = ev.parse_manifest(
            manifest(
                sample("low", bands(5, 5, 5, 5)),
                sample("mid", bands(6, 6, 6, 6)),
                sample("high", bands(8, 8, 8, 8), task_type="task1"),
            )
        )

    def report(self, *items, configs=None):
        runs = ev.parse_runs(run_lines(*items), self.samples)
        return ev.build_report(self.samples, runs, self.meta, configs)

    def test_report_per_config_with_failures_missing_essays_and_mismatches(self):
        report = self.report(
            {"sampleId": "low", "config": "m", "repeat": 0, "final": {**bands(6, 6, 6, 6), "overall": 6.0}},
            {"sampleId": "low", "config": "m", "repeat": 1, "final": {**bands(6, 6, 6, 7), "overall": 6.0}},
            {"sampleId": "mid", "config": "m", "repeat": 0, "final": bands(6, 6, 6, 6)},
            {"sampleId": "high", "config": "m", "repeat": 0, "error": "timeout"},
            {"sampleId": "high", "config": "other", "repeat": 0, "final": bands(8, 8, 8, 8)},
        )
        self.assertEqual([config["config"] for config in report["configs"]], ["m", "other"])
        m = report["configs"][0]
        self.assertEqual(m["coverage"]["essaysScored"], 2)
        self.assertEqual(m["coverage"]["essaysMissing"], ["high"])
        self.assertEqual(m["coverage"]["failedEvaluations"], 1)
        self.assertEqual(m["coverage"]["failureRate"], 0.25)
        # overall errors: +1.0, +1.5 (recorded 6.0 is ignored, criteria give 6.5), 0
        self.assertAlmostEqual(m["accuracy"]["overall"]["mae"], 2.5 / 3, places=4)
        self.assertAlmostEqual(m["accuracy"]["overall"]["withinHalfBand"], 1 / 3, places=4)
        self.assertEqual(m["overallMismatches"], ["low/repeat 1: recorded 6.0, criteria give 6.5"])
        self.assertEqual(m["repeatConsistency"]["essaysWithRepeats"], 1)
        self.assertEqual(m["accuracyByTask"]["task2"]["n"], 3)
        self.assertNotIn("task1", m["accuracyByTask"])
        self.assertEqual(report["references"]["byTask"], {"task1": 1, "task2": 2})

    def test_config_filter_rejects_unknown_names(self):
        item = {"sampleId": "mid", "config": "m", "final": bands(6, 6, 6, 6)}
        self.assertEqual(len(self.report(item, configs=["m"])["configs"]), 1)
        with self.assertRaises(ev.EvalInputError):
            self.report(item, configs=["nope"])

    def test_markdown_flags_synthetic_data(self):
        item = {"sampleId": "mid", "config": "m", "final": bands(6, 6, 6, 6)}
        runs = ev.parse_runs(run_lines(item), self.samples)
        report = ev.build_report(self.samples, runs, {"synthetic": True, "dataNotice": "note"})
        text = ev.render_markdown(report)
        self.assertIn("SYNTHETIC FIXTURES", text)
        self.assertIn("Data notice: note", text)
        self.assertIn("| m | 1 | 0 |", text)


class FixtureTest(unittest.TestCase):
    """The bundled data must stay self-written, short and clearly labelled."""

    def test_manifest_is_labelled_synthetic(self):
        raw = json.loads(ev.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        self.assertIs(raw["synthetic"], True)
        self.assertIn("仅用于测试框架", raw["dataNotice"])
        self.assertIn("分数不代表真实水平", raw["dataNotice"])
        for item in raw["samples"]:
            self.assertEqual(item["source"], "synthetic", item["id"])

    def test_bundled_essays_exist_and_are_short(self):
        samples, _ = ev.load_manifest(ev.DEFAULT_MANIFEST)
        self.assertEqual(ev.missing_essays(samples, ev.DEFAULT_MANIFEST.parent), [])
        essays = sorted((ev.DEFAULT_MANIFEST.parent / "essays").glob("*"))
        self.assertEqual({path.name for path in essays}, {Path(s.essay_ref).name for s in samples.values()})
        for path in essays:
            words = len(path.read_text(encoding="utf-8").split())
            self.assertLessEqual(words, 80, f"{path.name} is not a short synthetic sample")

    def test_runs_file_starts_with_a_notice(self):
        first = ev.DEFAULT_RUNS.read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("分数不代表真实水平", json.loads(first)["_notice"])

    def test_private_data_and_reports_are_git_ignored(self):
        ignored = (HERE / ".gitignore").read_text(encoding="utf-8").split()
        self.assertIn("private/", ignored)
        self.assertIn("reports/", ignored)


class CliTest(unittest.TestCase):
    def test_fixture_replay_writes_reports_and_tells_configs_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_out, markdown_out = Path(tmp) / "r.json", Path(tmp) / "r.md"
            with contextlib.redirect_stdout(io.StringIO()):
                code = ev.main(["--check-essays", "--json-out", str(json_out), "--markdown-out", str(markdown_out)])
            self.assertEqual(code, 0)
            report = json.loads(json_out.read_text(encoding="utf-8"))
            self.assertTrue(markdown_out.read_text(encoding="utf-8").startswith("# Writing scoring eval report"))
        self.assertTrue(report["synthetic"])
        configs = {config["config"]: config for config in report["configs"]}
        self.assertEqual(set(configs), {"fake-close", "fake-compressed"})
        close, compressed = configs["fake-close"], configs["fake-compressed"]
        self.assertLess(close["accuracy"]["overall"]["mae"], compressed["accuracy"]["overall"]["mae"])
        self.assertLess(compressed["distribution"]["spreadRatio"], close["distribution"]["spreadRatio"])
        self.assertGreater(
            compressed["distribution"]["predictedShare6to6_5"],
            compressed["distribution"]["referenceShare6to6_5"],
        )
        self.assertEqual(close["coverage"]["failedEvaluations"], 1)
        self.assertGreater(compressed["sampling"]["runsWithSamples"], 0)

    def test_bad_input_exits_with_code_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            path.write_text(json.dumps(manifest(sample("a", bands(6, 6, 6, 6.25)))), encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = ev.main(["--manifest", str(path)])
        self.assertEqual(code, 2)
        self.assertIn("6.25", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
