# Phi-4 BBH five-object deduction

All **250 BBH-five questions**, neutral and angry, **both original and explicit
reasoning prompts**: 1,000 answers. No vector collection, steering or filtering.
The prompts are identical except "Let's think step by step." versus "Give
concise step-by-step reasoning with only the necessary deductions." Both end
with the same `The answer is (<letter>).` requirement, with five valid choices.

```bash
git pull --ff-only
nohup bash run_bbh5_candidate.sh phi4 > outputs/bbh5-phi4.log 2>&1 < /dev/null &
tail -f outputs/bbh5-phi4.log
```

Waits for the model's GSM8K and BBH-three queues, current Kimi queues, and the
shared GPU lock. Reuses Phi-4's existing environment/checkpoint and all500
neutral/angry clips in `outputs/azure-kimi-bbh-five250/`. Missing clips are resumed
using compatible original Azure audio or generated with Azure variables if
needed. No processes are terminated; no artifacts are deleted. Existing Kimi
generations and vectors are neither modified nor used as Phi-4 responses.

Same native adapters as BBH-three/GSM8K, full audio, seed42, greedy decoding,
512-token budget. Input wording and gold targets are checked against the pinned
upstream BBH-five source checksum. Model inputs contain only audio plus the task
instruction. Phi-4's duration recommendation and all observed durations remain
documented in the shared adapter/config; no waveform is truncated.

```bash
.venv-diffa/bin/python evaluate_bbh5_candidates.py --model phi4 --mode summary
```

Reports use the installed upstream **five-object** strict/flexible filters and
exact-match metric. Additional format extractability checks all **ABCDE**
choices, not just ABC. Exact paired McNemar tests and95% paired-bootstrap
intervals use the same matched questions. Secondary tests are unadjusted;
prompt variants use the same cohort and are reported separately, not pooled.
This is spoken-adapted inference, not the full official text-only harness setup.

Outputs: `outputs/bbh-five-candidates/phi4/{original,explicit}/`, including raw
responses, frozen configs, model/package versions, audio hashes, token counts,
scorer provenance, rescored responses and summaries. Resume preserves completed
answers. Optional `--prompt original` / `--prompt explicit` selects one variant.
The same runner can later use `minicpm` or `flamingo`, with isolated outputs.
