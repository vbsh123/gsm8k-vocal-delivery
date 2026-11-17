# Auditing the SAME100 reasoning-request / emotion measurements

Run on the machine holding the saved states. CPU only; no model downloads,
forwards, generations, steering or edits to original experiment artifacts:

```bash
.venv-diffa/bin/python audit_reasoning_emotion.py --model diffa --check-audio
.venv-diffa/bin/python audit_reasoning_emotion.py --model phi4 --check-audio
```

Use `--output PATH` for another experiment folder. Omit `--check-audio` when
auditing downloaded state files without the original WAVs. The script requires
all 100 triples and fails if any are absent. It writes only a new
`independent-geometry-audit.json` in that experiment folder.

It checks frozen question/voice, audio hash receipts, exact condition prompts,
protocol, full layer/group/hidden-size shapes, finite values, native token-group
bounds and matched DIFFA answer-mask embeddings. It checks the combined DIFFA
audio mean against the length-weighted acoustic/semantic means.

From raw states it independently forms, in float64:

- `r = mean(neutral_reasoning - neutral_answer_only)`
- `e = mean(angry_reasoning - neutral_reasoning)`

Python `math.fsum` dot products then recompute cosine, coefficient, projected
squared norm, reverse sign, split-half reliability and both cross-half cosines.
These are checked against the saved vectors, CSV and summary. Projection checks
also verify that the residual is orthogonal to `r` and that squared component
norms sum to the original squared norm. The tolerance for float32 saved
statistics/vectors is 2e-5 absolute plus 2e-5 relative.

For `p = (<e,r>/<r,r>) r`, the coefficient has the sign of the projection,
while `||p||²/||e||² = cosine(e,r)²` loses that sign. This percentage is a
fraction of the mean emotion vector's squared norm, not explained accuracy,
variance across questions, or a percentage change in reasoning.

## Boundaries on interpretation

The collectors observe a reasoning-request contrast before answer generation,
not verified internal reasoning or reasoning effort. They use means of paired
differences, not mean per-question cosines. Large cosine/reliability can reflect
a stable prompt/style response shared across questions. Cross-half analysis
removes shared question-specific noise but cannot remove a common instruction
or token-position effect.

DIFFA averages 512 initial masked-answer positions. Phi captures one last valid
prompt position predicting the first answer token. These are native but
different measurements. The default DIFFA audio fallback uses a concise
reasoning request; Phi's original prompt uses "Let's think step by step."
Cohort selection also differs. Therefore opposite signs are descriptive
within-model observations, not a controlled demonstration that model training
alone causes the difference. Check the exact prompts and IDs in each saved
config/audit report. Audio length and instruction length can change token
positions; the audit records all three input lengths for each question.

The audit cannot retroactively prove runtime/checkpoint identity or native
generation parity on the GPU machine. DIFFA's shared loader is not revision
pinned. Preserve its runtime record, decoder source hash and experiment code.
Phi's checkpoint is revision pinned. No saved tensors can establish that the
overlapping component causes an accuracy change; intervention experiments
are needed for that claim.
