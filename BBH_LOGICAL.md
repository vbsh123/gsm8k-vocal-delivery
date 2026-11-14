# BBH logical deduction: 200 questions, four deliveries

This initial run uses the official **three-object** logical-deduction task (250
available questions). Five- and seven-object problems are longer and are less
suitable for DIFFA's 30-second audio input limit. Seed 42 shuffles the source once
and selects 200 questions, before generating audio or seeing model answers.

Source: `suzgunmirac/BIG-Bench-Hard`, revision
`9ee07bd481feebf959a6b59d61ea57bdcf30964d`; source checksum is pinned in code.
All original constraints, options and option order are retained. Only the markers
`(A)`, `(B)`, `(C)` become spoken `Option A:`, `Option B:`, `Option C:`. The target
answer is saved as metadata and never spoken or supplied to the model.

Azure Jenny reads each question in neutral, angry, sad and happy (cheerful) style,
intensity 1.5. Only questions with all four recordings at most 30 seconds enter
evaluation, without cropping. The realized count and excluded IDs are saved in
the evaluation config. These are 200 selected questions, not a guarantee that all
200 fit the limit. No outcome or classifier filtering.

**There is no steering, vector loading, or extraction in this script.**

DIFFA uses the existing shared generator, seed 42 reset for each recording,
512 output tokens, 128 diffusion steps, 32-token blocks, temperature zero and
fast-dLLM. It sees the audio and an identical generic problem/answer-format
instruction for every delivery, without a transcript or emotion label.

Reports separate answer accuracy from format compliance and strict accuracy,
using the existing custom `choice-relaxed-v2` parser against official BBH targets.
This is a zero-shot spoken adaptation, not the original BBH few-shot prompting
protocol or an official harness inference run. Neutral-paired differences include
McNemar p-values and 95% paired bootstrap intervals, unadjusted.

On the machine running SVAMP, with Azure credentials exported:

```bash
git pull --ff-only
mkdir -p outputs
nohup .venv-diffa/bin/python -u run_bbh_logical200.py \
  > outputs/bbh-logical200.log 2>&1 < /dev/null &
tail -f outputs/bbh-logical200.log
```

The launcher waits for the SVAMP process to exit **before generating or evaluating**.
Inference also takes the shared GPU lock and waits for active GPU work. A failed
SVAMP process counts as exited; inspect its log to distinguish completion from
failure. Rerunning resumes existing audio and completed answers.

Summary without GPU work, including during evaluation:

```bash
.venv-diffa/bin/python -u run_bbh_logical200.py --mode summary
```

Recordings and listening page: `outputs/azure-bbh-logical200/`.
Raw answers, eligibility/config and summaries: `outputs/bbh-logical200-diffa2-harness/`.

## Experimental unchanged long audio

The full BBH recordings proved too long for matched four-style native evaluation.
`--chunked-audio` reuses these exact WAVs without time stretching, cropping or TTS.
Whisper encodes nonoverlapping windows of at most 30 seconds. Valid hidden frames
from each encoder layer are concatenated chronologically, removing padded frames.
The existing semantic adapter and acoustic Q-former each run **once** on the
joined features, producing one semantic sequence and one 64-token acoustic summary.
Weights, text prompt, seed and fast-dLLM settings remain unchanged. Short audio
uses the original generator directly.

This is a new inference adaptation, not native trained long-context support.
Whisper positions restart in each window and encoder attention cannot cross
window boundaries. The downstream adapters see longer feature sequences. This
must be reported as experimental chunked inference.

First run two original questions in all four styles, with one model load:

```bash
nohup .venv-diffa/bin/python -u run_bbh_logical200.py \
  --mode evaluate --chunked-audio --smoke-test \
  > outputs/bbh-chunked-smoke.log 2>&1 < /dev/null &
tail -f outputs/bbh-chunked-smoke.log
```

The smoke output is separate: `outputs/bbh-logical200-diffa2-harness-chunked-smoke/`.
It checks identical generated responses on a short excerpt using the native and
wrapper paths, then saves responses for the same long recording with 20s/30s
boundaries in `chunking-preflight.json`. Boundary diagnostics aren't part of the
main accuracy table. The eight main smoke answers use 30-second windows.
Logs and answer rows include chunk counts, valid encoder frames and adapter token
counts. Inspect those responses and diagnostics before interpreting full results.

The full run uses the same 200 selected questions/all 800 original recordings:

```bash
nohup .venv-diffa/bin/python -u run_bbh_logical200.py \
  --mode evaluate --chunked-audio \
  > outputs/bbh-chunked.log 2>&1 < /dev/null &
```

Full answers go to `outputs/bbh-logical200-diffa2-harness-chunked/`. Smoke answers are
kept separate; a full run evaluates them again using the full-run configuration.
Use `--mode summary --chunked-audio` for the full summary, or add `--smoke-test`
for the smoke summary. Both use the pinned upstream BBH evaluator.

## Harness-aligned answer instruction

New runs explicitly request `The answer is (<letter>).`, with a parenthesized
A/B/C and final period, matching the upstream strict extractor. The text wrapper
includes the task description and “Let's think step by step” convention from the
zero-shot harness. It is still an audio adaptation, not the exact text prompt.
The instruction is identical across styles. Progress logs now show actual harness
strict/flexible correctness; the old custom Final-answer score remains a legacy
field only. Outputs use the `-harness` name to keep previous prompt runs separate.
Existing recordings are reused without any regeneration.
