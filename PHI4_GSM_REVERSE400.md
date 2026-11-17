# Phi-4 reverse anger direction on neutral GSM8K

```bash
git pull --ff-only
nohup bash run_phi4_gsm_reverse400.sh \
  > outputs/phi4-gsm-reverse400-400-L17.log 2>&1 < /dev/null &
tail -f outputs/phi4-gsm-reverse400-400-L17.log
```

Progress and CPU scoring, including during collection/evaluation:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u steer_phi4_gsm_reverse.py --mode summary
```

Uses the original1269 canonical GSM8K question bank, shuffled with seed42.
First400 are collection; next400 are disjoint test questions;469 are unused.
No response/outcome, duration, classifier or token-length filtering. This is
an exploratory follow-up: previous Phi outcomes and geometries were inspected.
Fresh means and baselines are computed; no outcome-selected regression subset.

All full recordings are reused from `outputs/azure-kimi-gsm8k1269`; the folder
name does not select the Kimi model. This runner uses only pinned
`microsoft/Phi-4-multimodal-instruct`, via existing `.venv-bbh-phi4`. No TTS,
Kimi checkpoint, DIFFA encoder, or custom audio chunking is used. `--source`
and `--output` override folders. Source audio must already be complete.

Uses the same ORIGINAL reasoning-requested Phi GSM prompt, including
"Let's think step by step." and the existing `#### <number>` suffix.
Native greedy generation: seed42 per recording/condition, BF16/SDPA, KV cache,
max512 new tokens, one beam, repetition penalty1. Baseline has no hook.

## Vector collection

On400 collection questions, generate neutral and angry answers (800 native
generations). At zero-based decoder layer17 (the eighteenth block), capture
the last position that predicts the next answer token on EVERY native forward.
This includes the last valid prompt position on prefill and the last cached
token position on subsequent forwards. Average equally over forwards within
each generation, form `angry_mean - neutral_mean`, then average equally across
questions. Save original responses, per-style means/counts and per-pair
differences. The two styles use their own generated histories, matching our
usual native mean-vector method, not teacher-forced/matched-state differences.

This is a NEW Phi direction collected on these400 questions, not the earlier
100-question initial-forward-only reasoning/emotion diagnostic. Its geometry
need not match that diagnostic. No normalization or strength tuning occurs.

## Evaluation

Generate three answers per held-out question (1200 generations): neutral,
neutral with `-0.5 * (angry - neutral)`, and neutral with `-1 * (angry - neutral)`.
Add only to L17's last answer-prediction position on every forward. Audio-prefix
positions are not directly changed. Each generation starts with a fresh KV
cache; hook cleanup is guaranteed even on failure. No angry test generation
or positive-strength intervention is added to this experiment.

This operationally moves away from the collected angry-associated direction;
it does not guarantee "more neutral", more reasoning, or better accuracy.

Outputs in `outputs/phi4-gsm-reverse400-400-L17`: frozen `run_config.json`,
runtime,400 small `pairs/*.pt`, `style-vectors.pt`, raw `results.jsonl`, scorer
provenance, rescored answers and `summary.json`. About20MiB collection data,
plus raw text. Atomic pair caches and completed evaluation answers resume;
the fixed split never grows. Interrupted final JSONL rows may be repaired only
by the evaluation writer. CPU reports read without taking the GPU/job lock.

Scoring uses the installed EleutherAI GSM8K task API's original strict and
flexible filters/metric, with numeric equivalence as an additional primary
metric. Format extractability is a separate diagnostic. Audio-adapted prompts
are not full official text-harness inference. Paired comparisons use questions
completed in ALL THREE conditions, exact McNemar p-values and paired bootstrap
95% intervals, unadjusted. Generated suffix token counts include any emitted
EOS, as in the earlier native Phi evaluation.
