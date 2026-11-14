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
Raw answers, eligibility/config and summaries: `outputs/bbh-logical200-diffa2/`.
