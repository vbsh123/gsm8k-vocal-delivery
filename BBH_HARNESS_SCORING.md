# BBH upstream scoring

`run_bbh_logical200.py` now uses EleutherAI lm-evaluation-harness BBH CoT zero-shot
answer filters and its exact-match metric for its final summary. Existing raw
generations, audio, prompts, targets and decoding are unchanged. Per-generation
progress fields remain explicitly labeled custom diagnostics; the final harness
summary is the evaluator result.

This uses the standard EleutherAI harness implementation, not an evaluator
published by the BBH paper authors. It is an **audio-adapted evaluation with
upstream scoring**, not the official text/few-shot evaluation protocol.

The runner uses both upstream task pipelines, as configured in:
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/bbh/cot_zeroshot/logical_deduction_three_objects.yaml

- `flexible-extract`: upstream BBH MultiChoiceRegexFilter, including its fallback
  rules for option text and colon-prefixed letters, then upstream TakeFirstFilter.
- `strict-match`: upstream RegexFilter, then TakeFirstFilter. It expects phrases
  such as `The answer is (B).`, not our existing `Final answer: B` instruction.
- Both use the upstream exact_match function and normalization from the task's
  included template, against the original `(A)`/`(B)`/`(C)` target.

No responses are reformatted to make them pass. No stop sequences are applied
retroactively. Flexible extraction may pick choices mentioned in reasoning: that
is the upstream behavior, not a custom correction. Strict results can be low
because our generation prompt requests a different final marker. A separate
prompt-format experiment would require new generations and a new output folder.

## Rescore without rerunning the model

```bash
cd /workspace/gsm8k-vocal-delivery
source .venv-diffa/bin/activate
python score_bbh_harness.py \
  --source outputs/azure-bbh-logical200 \
  --output outputs/bbh-logical200-diffa2
```

The usual summary command also uses this scorer:

```bash
python run_bbh_logical200.py --mode summary
```

No GPU, Azure key, model downloads or new model environment are needed. The first
call fetches a few small upstream source/config files. Their Git revision and
SHA256 hashes are saved under `outputs/bbh-logical200-diffa2/harness/upstream/`.
Existing caches are checked for source changes on subsequent calls.

To avoid installing the full harness and its model dependencies, the adapter
executes selected upstream AST definitions: RegexFilter, TakeFirstFilter,
ExtendedRegexFilter, MultiChoiceRegexFilter and exact_match_hf_evaluate. Their
bodies are unchanged; registration decorators and abstract Filter inheritance
are omitted. Configuration is read from the pinned upstream YAML, not manually
recreated. The upstream license is retained with the downloaded files.

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
