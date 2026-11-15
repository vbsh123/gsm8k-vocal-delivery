# BBH upstream scoring

`run_bbh_logical200.py` now uses EleutherAI lm-evaluation-harness BBH CoT zero-shot
answer filters and its exact-match metric for its final summary. Existing raw generations and audio remain unchanged. New runs request the
upstream-compatible answer phrase and use separate `-harness` output folders.
Per-generation progress and final summaries report actual harness scores.

This uses the standard EleutherAI harness implementation, not an evaluator
published by the BBH paper authors. It is an **audio-adapted evaluation with
upstream scoring**, not the official text/few-shot evaluation protocol.

The runner uses both upstream task pipelines, as configured in:
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/bbh/cot_zeroshot/logical_deduction_three_objects.yaml

- `flexible-extract`: upstream BBH MultiChoiceRegexFilter, including its fallback
  rules for option text and colon-prefixed letters, then upstream TakeFirstFilter.
- `strict-match`: upstream RegexFilter, then TakeFirstFilter. It expects phrases
  such as `The answer is (B).`, the new requested answer format. Earlier runs requested `Final answer: B`.
- Both use the upstream exact_match function and normalization from the task's
  included template, against the original `(A)`/`(B)`/`(C)` target.

No responses are reformatted to make them pass. No stop sequences are applied
retroactively. Flexible extraction may pick choices mentioned in reasoning: that
is the upstream behavior, not a custom correction. Earlier Final-answer runs can have zero strict accuracy despite correct answers.
New runs explicitly request the upstream phrase; this requires new generations,
using the same saved audio and a separate output folder.

## Rescore without rerunning the model

```bash
cd /workspace/gsm8k-vocal-delivery
source .venv-diffa/bin/activate
bash setup_harness.sh
python score_bbh_harness.py \
  --source outputs/azure-bbh-logical200 \
  --output outputs/bbh-logical200-diffa2
```

The usual summary command also uses this scorer:

```bash
python run_bbh_logical200.py --mode summary
```

No GPU, Azure key or model downloads are needed. Install the pinned harness once
with `setup_harness.sh`. Scoring uses the installed task's `apply_filters()` and
`process_results()` methods. No AST definitions are extracted or rewritten.
Installed task source hashes and version information are saved in
`harness/upstream/api-provenance.json`.

To compare the completed three-object delivery/steering runs and SVAMP against
their prior summaries without overwriting those summaries, use
`audit_harness_api.py`; see `HARNESS_API.md`.

Outputs are `harness/summary.json` and `harness/rescored.jsonl`. Original results
are untouched. All four deliveries must be complete for a question/voice to enter
the accuracy tables and neutral-versus-emotion paired tests. Source checksums and
each selected question's original input/target are validated before scoring.

## Tests against an actual cached upstream revision

```bash
BBH_HARNESS_CACHE=outputs/bbh-logical200-diffa2/harness/upstream \
  python -m unittest discover -s tests -p test_bbh_logical200.py -v
```

Fixtures distinguish phrase-sensitive strict extraction from flexible extraction,
verify four-way matching and check that saved responses are not modified.
