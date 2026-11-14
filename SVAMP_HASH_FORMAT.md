# SVAMP answer-format control

Run the same frozen SVAMP questions and existing neutral/angry audio again,
changing only the requested final-answer marker from `Final answer: <number>`
to `#### <number>`. No new Azure audio, extraction, steering or outcome filtering.
Seed42, 512 answer positions, native fast_dllm, 128 steps, block32 and temperature0
are unchanged. Because generation is rerun, reasoning and numeric answers can
change too. This tests a prompt-format effect, not merely rescoring old strings.

## Run on the existing Vast machine

```bash
cd /workspace/gsm8k-vocal-delivery
source .venv-diffa/bin/activate
export PYTHONDONTWRITEBYTECODE=1
python -u run_svamp500.py --mode evaluate --unsteered-only \
  --source outputs/azure-svamp500 --answer-format hashes \
  --output outputs/svamp500-diffa2-hashes
```

`evaluate` reuses saved WAVs and does not invoke Azure. It waits for the shared
GPU lock and active GPU jobs as the original SVAMP runner does. The new output
folder supports resume. Reusing the original run folder with a different prompt
is rejected. Hash-format mode rejects steering because the frozen vector was
extracted using the original prompt.

## Read the results

All accuracy tables and paired tests use completed neutral/angry pairs matched
by question and voice. The report includes:

- `flexible_numeric`: upstream GSM8K flexible extraction plus our numeric equality;
  remains the primary metric in both prompt conditions.
- `harness_flexible`: upstream flexible extraction plus normalized string match.
- `strict`: our original `Final answer:` scorer; a separate format diagnostic.
- `harness_strict`: upstream RegexFilter using `#### (\-?[0-9\.\,]+)`, selecting
  the first match, followed by normalized string exact match. For example `51.00`
  does not equal `51` under this string metric, though both are numerically equal.
- `harness_format_compliant`: a `####` numeric answer was extractable, regardless
  of correctness. This is not proof the entire response followed the instruction.

The strict regex and normalization follow the GSM8K harness configuration:
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml
The runner pins/caches the actual upstream RegexFilter and records its source
provenance in each output's `upstream/` folder. This is not a full official harness
inference evaluation: audio, prompt, model and decoding remain our setup.

Rescore the original saved responses with the additional columns, without GPU:

```bash
python run_svamp500.py --mode summary --unsteered-only \
  --source outputs/azure-svamp500 --output outputs/svamp500-diffa2
```

Resummarize the new run:

```bash
python run_svamp500.py --mode summary --unsteered-only --answer-format hashes \
  --source outputs/azure-svamp500 --output outputs/svamp500-diffa2-hashes
```

Interpret strict accuracy together with extractable-format rate and flexible
numeric accuracy. An increase only in strict accuracy can reflect formatting;
an increase in flexible numeric accuracy indicates the extracted numeric answers
changed favorably. Compare both delivery gaps rather than choosing the best
prompt separately for each emotion. Until both runs finish, their available
matched subsets may differ; final comparisons should use the same question IDs.
