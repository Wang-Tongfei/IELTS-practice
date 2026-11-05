# Writing eval

Measures how close the writing scores are to known bands and how stable they
are. Follows the `coach-eval` convention: without the `WRITING_EVAL_*`
variables nothing touches the network; with them, `--record` scores the
manifest with a real model using the application's real request and rules.

```bash
# Bundled synthetic fixtures: validate and report (no network)
python developer/tests/tools/writing-eval/eval_writing.py --check-essays

# Format check only (what CI needs)
python developer/tests/tools/writing-eval/eval_writing.py --validate-only

# Unit tests (run by developer/tests/ci/run_static_suite.py)
python -m unittest discover -s developer/tests/tools/writing-eval -p "test_*.py"
```

## Recording a real model

```bash
export WRITING_EVAL_BASE_URL=https://api.deepseek.com/v1   # ".../chat/completions" is appended
export WRITING_EVAL_API_KEY=...                             # never commit it
export WRITING_EVAL_MODEL=deepseek-chat

python developer/tests/tools/writing-eval/eval_writing.py \
  --manifest developer/tests/tools/writing-eval/private/manifest.json \
  --record developer/tests/tools/writing-eval/private/runs-deepseek.jsonl \
  --repeats 2 --label deepseek-chat/builtin
```

What one evaluation costs and does, matching `writing_evaluation.rs`:

1. The request comes from the Rust crate (test hook
   `writing_eval_hook_when_asked`): built-in rubric for the task type,
   `SCORING_CONTRACT`, measured facts, temperature, output ceiling and cache key.
   Needs cargo; the first run compiles `ielts-application`.
2. Two scoring calls. A failed call is recorded as `null` and dropped, as the
   app does.
3. The raw replies are replayed through `collect_samples` and
   `aggregate_samples`: half-band snapping, ceilings, median. When the two
   disagree by a band or more, one more call is made and replayed, exactly
   where the app would make it.
4. The review (feedback) call is not made: it does not change the score.

So `--record` costs 2-3 scoring calls per essay per repeat. Requests are sent
one at a time (the app sends the first two concurrently, which only matters
for prompt caching).

The request body mirrors the gateway in `src-tauri/src/ai/runtime.rs`:
o1/o3/o4/gpt-5 models get `max_completion_tokens` (at least 1024) and no
`temperature`; `api.openai.com` gets `prompt_cache_key` and `openrouter.ai`
gets `session_id`. Other endpoint quirks of the app's gateway (retries,
context budget, truncation detection) are not reproduced.

Each run keeps the raw `replies`, the `config` (default
`<model>/<prompt fingerprint>`, the fingerprint being a hash of the system
prompts actually sent) and the `ceilings` that applied.

## Re-scoring without new calls

After changing post-call rules (ceilings, rounding, the third-sample
threshold), replay an old recording through the current code:

```bash
python developer/tests/tools/writing-eval/eval_writing.py \
  --manifest developer/tests/tools/writing-eval/private/manifest.json \
  --rescore developer/tests/tools/writing-eval/private/runs-deepseek.jsonl \
  --out developer/tests/tools/writing-eval/private/runs-deepseek-rules2.jsonl --label rules2
```

Then compare both files in one report:
`--runs <old.jsonl> --runs <new.jsonl>`. A change to the prompt itself needs a
new `--record`, since the replies depend on it. If a rule change makes the app
want a third call that the recording lacks, the run is marked `needsThird`
and scored from the first two.

## Data policy

- `fixtures/` holds only short texts written for this repository, with
  invented bands and task prompts. **仅用于测试框架，分数不代表真实水平。**
  `FixtureTest` fails if a bundled sample is not labelled `synthetic` or grows
  past 80 words.
- Copyrighted essays (Cambridge books, official sample scripts, paid course
  material, essays learners did not license) never go into this repository,
  the app bundle or a PR description. Keep them in `private/` (git-ignored) or
  outside the tree and pass the path with `--manifest` / `--runs`.
- Recordings keep the raw replies, which can quote the essays: they belong
  in `private/` too. Every recording starts with a `_notice` line saying so.
- `reports/` is git-ignored as well.

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
      "taskPrompt": "Some people think ... To what extent do you agree or disagree?",
      "reference": {"taskResponse": 6.0, "coherence": 6.5, "lexical": 6.0, "grammar": 5.5},
      "referenceOverall": 6.0
    }
  ]
}
```

`essayRef` is relative to the manifest. It may be an opaque id for reporting
only, but `--record`, `--rescore` and `--check-essays` need a readable file.
`taskPrompt` is optional; without it the model sees "Task prompt: not
provided", as in the app. The reference overall is derived from the four
criteria with the application's rule (mean of four, `.25` and `.75` round up);
`referenceOverall` is optional and only checked.

## Runs (JSONL, one evaluation per line)

```json
{"sampleId": "s-001", "config": "deepseek-chat/builtin", "repeat": 0,
 "final": {"taskResponse": 6.0, "coherence": 6.0, "lexical": 6.0, "grammar": 6.0, "overall": 6.0},
 "samples": [{"taskResponse": 6.0, "coherence": 6.0, "lexical": 6.0, "grammar": 6.0},
             {"taskResponse": 6.5, "coherence": 6.0, "lexical": 6.0, "grammar": 6.0}],
 "replies": ["{...raw reply...}", "{...raw reply...}"]}
{"sampleId": "s-001", "config": "deepseek-chat/builtin", "repeat": 1, "error": "recorded failure", "replies": [null, null]}
```

`final` is what the learner would see; a recorded `overall` that disagrees
with the criteria is listed under `overallMismatches`. `samples` are the
scoring calls after ceilings. Lines with a `_notice` key and blank lines are
skipped.

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
