# MiniCPM GSM8K same100 diagnostics, concurrent with Phi

After freeing the unused Kimi model cache:

```bash
git pull --ff-only
nohup bash run_minicpm_gsm100_diagnostics.sh \
  > outputs/minicpm-gsm100-diagnostics.log 2>&1 < /dev/null &
tail -f outputs/minicpm-gsm100-diagnostics.log
```

No shared GPU lock and no active-job wait. Only this experiment's own duplicate
job lock is used. Default Torch allocator cap22GiB; at least24GiB free is required
at model startup, otherwise exit immediately rather than wait or stop Phi.
The cap excludes non-Torch CUDA allocations; concurrent workloads can still
fail, so this is a resource bound, not a guarantee against OOM. No quantization
or offload, no changes to the running Phi process. Both experiment caches resume.
Concurrent generation times are not isolated speed measurements.

Pinned MiniCPM-o4.5, native BF16/SDPA audio inference, vision/TTS disabled.
Separate `.venv-bbh-minicpm`, using the already prepared setup script and model
revision in `bbh_candidate_models.json`. Initial disk estimate includes missing
17.46GiB weights, up to6GiB environment, plus4.5GiB reserve/results. No automatic
deletions, no new TTS. Only the checkpoint selected by the script is downloaded.

Exactly the SAME100 GSM8K question/voice pairs from
`outputs/phi4-reasoning-emotion100-gsm8k/run_config.json`, no re-selection or
outcome filtering. All three conditions share those100 questions:

1. Neutral audio with the original reasoning-request prompt.
2. Same neutral audio with the original answer-only prompt.
3. Angry audio with the SAME reasoning-request prompt.

300 full native greedy text generations, seed42, max512, one beam, repetition
penalty1. The native `enable_thinking=False` setting is held fixed across ALL
conditions; the contrast changes the textual reasoning request, not a model
mode. Content ordering follows MiniCPM's native adapter: text instruction, then
audio in a user message; no gold/question transcript/emotion label is supplied.
Its template/order differs from Phi. Full recordings enter native processing.

## Captures and diagnostics

At all36 decoder layers,4096 dimensions:

- INITIAL forward: last valid prompt state predicting first answer token, plus
  mean over native audio feature positions from `get_omni_embedding.audio_bounds`.
- GENERATION mean: last answer-prediction position on every native forward,
  averaged equally within each generation. Audio-prefix states are captured
  only on prefill, as subsequent cached forwards do not contain those positions.

Initial reasoning direction = mean(neutral_reasoning - neutral_answer_only).
Initial emotion direction = mean(angry_reasoning - neutral_reasoning).
Those contrasts are also saved separately for whole-generation answer means.
The initial direction is comparable in METHOD to Phi's pre-generation diagnostic;
the whole-generation direction additionally reflects divergent generated text.
No steering or claimed mechanism is inferred from cosine alone.

Outputs `outputs/minicpm-gsm100-diagnostics`: frozen config, runtime,300 small
resumable states with raw responses, `results.jsonl`, `style-vectors.pt`,
`layer-diagnostics.csv`, `summary.json`, harness provenance, `rescored.jsonl`,
`accuracy-summary.json`. Partial summaries contain complete triples only and
write `style-vectors-partial.pt`. Initial and whole-generation vectors are
explicitly distinct. Expected state data about500MiB, no model copies.

All-layer initial-state cosine, signed projection, squared-norm overlap,
cross-half checks and relative angry/neutral drift are reported for answer and
audio positions. Reasoning-instruction overlap is not proven reasoning effort;
prompt style/positions and shared-reference effects can contribute. Full-model
comparison requires those controls. No geometry significance tests claimed.

```bash
# CPU summaries can run while collection continues, from completed triples.
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u diagnose_minicpm_gsm100.py --mode summary
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u diagnose_minicpm_gsm100.py --mode score
```

Upstream installed GSM8K task filters/metric are applied to saved native text
answers; numeric equivalence is additional. Audio-adapted inference, not full
official text-harness prompting. Token lengths are RETOKENIZED response text,
unlike Phi's generated suffix count including EOS. Compare lengths within a
model, not exact counts across tokenizer families. Accuracy p-values/CIs are
paired, exploratory and unadjusted. Never compare partial cohorts as independent.
