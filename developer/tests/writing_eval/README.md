# Writing scoring eval harness

Offline report on how close the writing scores are to known bands and how
stable they are. It never calls a model and needs no network or API key: it
replays recorded (or fabricated) scores against a sample manifest.

```bash
# Synthetic fixtures: exercises the harness only
python developer/tests/writing_eval/run_writing_eval.py --check-essays

# Unit tests (also run by developer/tests/ci/run_static_suite.py)
python -m unittest discover -s developer/tests/writing_eval -p "test_*.py"

# A private sample set (git-ignored directory)
python developer/tests/writing_eval/run_writing_eval.py \
  --manifest developer/tests/writing_eval/private/manifest.json \
  --runs developer/tests/writing_eval/private/runs-modelA.jsonl \
  --runs developer/tests/writing_eval/private/runs-modelB.jsonl \
  --json-out developer/tests/writing_eval/reports/report.json \
  --markdown-out developer/tests/writing_eval/reports/report.md
```

## Data policy

- `fixtures/` holds only short texts written for this repository, with
  invented bands. **仅用于测试框架，分数不代表真实水平。** `FixtureTest`
  fails if a bundled sample is not labelled `synthetic` or grows past 80 words.
- Copyrighted essays (Cambridge books, official sample scripts, paid course
  material, essays learners did not license to us) never go into this
  repository, the app bundle or a PR description. Keep them in `private/`,
  which is git-ignored, or anywhere outside the tree, and pass the path with
  `--manifest` / `--runs`. `essayRef` may be an opaque id (for example
  `book-17/test-2/task-2`) instead of a path; the harness only reads essays
  when `--check-essays` is given.
- `reports/` is git-ignored too: a report on private data can quote ids that
  identify the source.

## Manifest (`schemaVersion: 1`)

```json
{
  "schemaVersion": 1,
  "synthetic": false,
  "dataNotice": "who wrote and rated these essays, and under what licence",
  "samples": [
    {
      "id": "s-001",
      "taskType": "task2",
      "essayRef": "essays/s-001.txt",
      "source": "two certified raters, adjudicated",
      "reference": {"taskResponse": 6.0, "coherence": 6.5, "lexical": 6.0, "grammar": 5.5},
      "referenceOverall": 6.0
    }
  ]
}
```

`reference` holds the four criterion bands (0-9, half bands). The reference
overall is derived from them with the application's rule (mean of four,
`.25` and `.75` round up); `referenceOverall` is optional and only checked.

## Runs (JSONL, one evaluation per line)

```json
{"sampleId": "s-001", "config": "deepseek-chat/builtin-rubric-v2", "repeat": 0,
 "final": {"taskResponse": 6.0, "coherence": 6.0, "lexical": 6.0, "grammar": 6.0, "overall": 6.0},
 "samples": [{"taskResponse": 6.0, "coherence": 6.0, "lexical": 6.0, "grammar": 6.0},
             {"taskResponse": 6.5, "coherence": 6.0, "lexical": 6.0, "grammar": 6.0}]}
{"sampleId": "s-001", "config": "deepseek-chat/builtin-rubric-v2", "repeat": 1, "error": "timeout"}
```

- `config` names everything that can change a score: provider, model, prompt
  version, sampling strategy. Each config is reported separately.
- `final` is what the learner saw (`WritingScoreV4`). A recorded `overall`
  that disagrees with the criteria is listed under `overallMismatches`.
- `samples` (optional) are the individual scoring calls behind `final`,
  after ceilings, so the report can show the third-call rate and how often
  the first two calls split by half a band.
- `repeat` distinguishes repeated evaluations of the same essay; run every
  essay the same number of times when comparing configs.
- Lines with a `_notice` key and blank lines are skipped.

## Metrics

Every successful evaluation counts once.

| Section | Metric | Meaning |
|---|---|---|
| `accuracy.<field>` | `mae`, `bias`, `rmse` | error against the reference; `bias` > 0 means scores run high |
| | `exact`, `withinHalfBand`, `withinOneBand` | share of evaluations within 0 / 0.5 / 1 band |
| | `qwk`, `pearson` | agreement beyond chance (quadratic weighted kappa over the 19 half bands) and linear correlation |
| `distribution` | `predictedShare6to6_5` vs `referenceShare6to6_5` | whether scores bunch on 6-6.5 |
| | `spreadRatio` | std of predicted overall / std of reference overall; below 1 means compressed |
| | `byReferenceBand` | bias per reference band; positive at the bottom and negative at the top is regression to the middle |
| `repeatConsistency` | `overallIdentical`, `meanPairDifference`, `spreadWithinHalfBand` | how much repeated evaluations of the same essay move |
| `sampling` | `meanScoringCalls`, `thirdSampleRate`, `firstPairHalfBandSplitRate` | cost and the share of averages that are snapped upward |
| `coverage` | `failureRate`, `essaysMissing` | evaluations that produced no score |

The harness reports; it does not gate. There is no pass/fail threshold until a
real, licensed sample set exists.

## Producing real runs

Not automated yet. The app stores each evaluation with `provider_id`, `model`,
`prompt_version` and the final score; scoring samples are not stored. A
recorder that drives `WritingEvaluationService` over a manifest with a real
model (and keeps each scoring sample) is a follow-up, run by hand with the
user's own key.
