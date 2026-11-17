# Same100 Phi-4 reasoning-instruction / emotion collection

Exactly **100 question/voice pairs**, with the SAME100 in all three conditions:

1. Neutral audio, original reasoning-requested prompt.
2. That neutral audio, answer-only prompt.
3. Matched angry audio, original reasoning-requested prompt.

The neutral-reasoning forward is reused for BOTH contrasts. No second set of100
is selected. This is **300 native prefills**, not300 full generated answers.

```bash
git pull --ff-only
nohup bash run_phi4_reasoning_emotion100.sh \
  > outputs/phi4-reasoning-emotion100.log 2>&1 < /dev/null &
tail -f outputs/phi4-reasoning-emotion100.log
```

Default task is GSM8K. It selects100 seed42-shuffled completed pairs from
`outputs/phi4-answer-only-followup/gsm8k/original/`, falling back to the original
Phi-4 GSM8K run if the follow-up baseline doesn't exist. No outcome, classifier,
format or length filtering. The bank and hash receipts are frozen before model
loading. Resuming never changes the selected questions. Pass `--source-run`
for an explicit original-prompt run folder. Optional `--task bbh3` runs the same
100-question design on BBH-three instead, in its own default output folder.

The queue waits for current answer-only, GSM8K and BBH Phi-4 jobs, then acquires
the shared GPU lock and waits for other GPU work to finish. No job is stopped.
Only the existing Phi-4 environment and pinned checkpoint are used. No TTS or
additional checkpoint is needed. All full audio, model processors, seed42 and
SDPA/bfloat16 settings match the original evaluation route.

## What is collected

Every one of the32 language decoder blocks saves its output at the last valid
prompt position, which predicts the first answer token, **before generation**.
No different generated histories or answer lengths can enter these contrasts.
Use `use_cache=False` and compute only the first-answer logits; no state is
reused between conditions. Audio-token means are also saved from those blocks.
No interventions or generation accuracy evaluation occurs in this collector.

Equal means across the same100 per-question differences:

- `reasoning_minus_answer_only`: neutral-reasoning − neutral-answer-only.
- `angry_minus_neutral`: angry-reasoning − neutral-reasoning.
- `neutral_minus_angry`: exact negative of angry-minus-neutral.

The reverse is saved explicitly, with no redundant model forwards. Its cosine
must be the opposite sign; it isn't independent supporting evidence. The
instruction contrast identifies a **reasoning-request representation**, not
established internal reasoning. It can reflect instruction wording or expected
output style too.

## Comparison

```bash
.venv-diffa/bin/python collect_phi4_reasoning_emotion.py --mode summary
```

For each layer: signed cosine for both emotion orientations, norms, projection
coefficient, squared cosine overlap, and split-half reproducibility. Because both
contrasts subtract/share the same neutral-reasoning state, same-question cosine
can be influenced by that shared baseline. The CSV also compares reasoning
estimated on odd selected questions with emotion on even ones, and vice versa,
using the same original100 and no new forwards. Those are descriptive checks,
not independent validation or adjusted significance tests.

Audio tokens precede the text instruction in Phi-4's native template, so causal
attention prevents them from reading the reasoning request. Their instruction
contrast should be near zero; its norm is recorded. Primary cosine analysis
therefore uses first-answer prediction states rather than interpreting a near-
zero audio-instruction vector's angle.

Outputs: `outputs/phi4-reasoning-emotion100-gsm8k/` with frozen config, runtime,
resumable per-question/per-condition state files, `style-vectors.pt`,
`layer-cosines.csv`, and `summary.json`. Roughly250MB of state caches, no full
encoded inputs or model weights. Incomplete summaries write
`style-vectors-partial.pt`; the final vector file requires all100 matched triples.
Raw answer-only and original evaluation results remain untouched.
