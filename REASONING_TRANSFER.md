BBH-five transfer, SVAMP mean collection, and aligned GSM8K evaluation
==================================================================

Run on the steering machine that has the completed BBH, SVAMP and GSM8K data.
No original extraction files, vectors, answers or audio are changed.

```bash
git pull --ff-only
bash setup_harness.sh
nohup .venv-diffa/bin/python -u run_reasoning_transfer.py \
  > outputs/reasoning-transfer.log 2>&1 < /dev/null &
tail -f outputs/reasoning-transfer.log
```

Export `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` in that shell first. Azure
generates 500 BBH-five clips and 500 new neutral SVAMP clips; model inference
uses the GPU. Existing recordings are reused and missing TTS files resume.

The stages run sequentially:

| Stage | Collection | Evaluation | Model generations before resume |
|---|---|---|---:|
| `bbh5` | None; existing BBH-three mean | All 250 five-object questions, neutral / angry / steered neutral | 750 |
| `svamp` | Existing 500 neutral/angry pairs | Other 500 official SVAMP questions, neutral / steered neutral | 1,000 collection + 1,000 evaluation |
| `gsm8k` | None; existing original GSM8K mean | Same 718 neutral recordings, plus their angry versions and steered neutral; fresh aligned-prompt generations | 2,154 |

Total: 4,904 model generations before resuming any completed work. Only SVAMP
collects a new vector. It is the usual angry-minus-neutral mean, not an optimized
vector or predictor: average token positions within each L31 forward, retain the
matched neutral/angry forward count per question, average the differences, then
average equally across the 500 extraction questions. The test half contains the
other 500 questions from the same fixed seed42 shuffle of the official source.
There is no outcome or emotion-classifier filtering.

All steering is **L31 answer tokens, strength 1, always on**. The professor's
strength sweep is a separate future sensitivity experiment, not included here.

Prompt and evaluator
--------------------

- BBH uses the previous prompt with choices changed from A–C to A–E, requesting
  `The answer is (<letter>).` The existing three-object vector is unchanged;
  failure does not automatically trigger any new BBH collection.
- SVAMP collection and evaluation both use the completed SVAMP `#### <number>`
  prompt. The original 500 pairs are unchanged. Only neutral audio is generated
  for the new 500 test questions.
- GSM8K uses the aligned `#### <number>` prompt in **all three conditions**.
  Baselines, angry answers and steered answers are generated fresh. The original
  vector, collected under the old prompt, is unchanged: this is also a test of
  transferring that vector to a new output-format instruction.
- Installed harness task APIs apply BBH strict/flexible filters and exact-match metrics from the upstream
  EleutherAI five-object task. GSM8K uses upstream GSM8K filters and exact-match
  metric. SVAMP uses those same GSM8K filters/metric on its numeric references;
  this is not a dedicated official SVAMP inference task. Numeric equivalence is
  an additional, explicitly separate metric for SVAMP and GSM8K. Marker
  extraction/compliance is also reported separately.
- These are audio-adapted generations scored with upstream code, not a full
  official text-harness inference protocol. They do not introduce text few-shot
  examples or change our decoder to the upstream generation/stopping protocol.

Seed42 resets before every generation. Decoder settings remain fast_dllm,
temperature 0, 128 steps, 32-token blocks and 512 answer tokens. The shared
`install_interventions` function applies steering. Only one DIFFA model is loaded
per stage; SVAMP collection and its evaluation use the same loaded model. Each
stage runs in a separate subprocess that exits before the next loads a model.

Full audio is retained. Clips up to 30 seconds follow the unchanged native
generation path. Longer clips use the experimental 30-second chunked Whisper
encoding followed by one global adapter pass, as in our completed BBH run.
In particular, a few angry GSM8K recordings may exceed 30 seconds although
neutral fits; they are encoded in full to preserve the fixed 718-question cohort.
Each result records its actual encoding mode; there is no silent crop or skip.

Paths and progress
------------------

Expected existing inputs:

- BBH vector: `outputs/bbh-logical250-angry-mean/style-vectors.pt`.
- SVAMP: `outputs/azure-svamp500/{SVAMP-source.json,questions.jsonl,plan.jsonl}`
  and its neutral/angry WAV files.
- GSM8K cohorts: `outputs/steer-jenny-late-layers/original_mean/run_config.json`
  and `outputs/steer-jenny-late-layers/third-cohort/original_mean/run_config.json`.
  The original GSM8K vector is taken from the cohort config.
- GSM8K audio plans/WAVs: `outputs/azure-jenny-next500` and
  `outputs/azure-jenny-remaining-test`. Neutral hashes must match the original
  steering recordings; angry recordings must have the same question and voice.

New audio:

- `outputs/azure-bbh-five250`
- `outputs/azure-svamp-remaining500`

New experiment outputs: `outputs/reasoning-transfer/{bbh5,svamp,gsm8k}/`.
Each contains its frozen `run_config.json`, raw `results.jsonl`, `summary.json`,
`rescored.jsonl`, and pinned harness source under `harness/upstream/`. SVAMP also
contains resumable extraction `pairs/` and `style-vectors.pt`. The queue writes
`outputs/reasoning-transfer/status.json` with the current stage/status.

Check every completed result without loading the model:

```bash
.venv-diffa/bin/python run_reasoning_transfer.py --stage summary
cat outputs/reasoning-transfer/status.json
```

The summary includes per-condition accuracy and paired gains, McNemar exact
p-values and 95% paired bootstrap intervals, all unadjusted. Numeric equivalence
is primary for math tasks; flexible extraction is primary for BBH. Existing
GSM8K development results remain exploratory under the new prompt too.

To run or resume just one stage:

```bash
.venv-diffa/bin/python -u run_reasoning_transfer.py --stage bbh5
.venv-diffa/bin/python -u run_reasoning_transfer.py --stage svamp
.venv-diffa/bin/python -u run_reasoning_transfer.py --stage gsm8k
```

The default queue waits for active SVAMP, BBH, finalist and Azure-emotion jobs,
then takes the shared GPU lock and waits for GPU availability. Saved answers
and extraction pairs are reused individually after interruption. Use `--help`
to override existing input paths if files were moved.

## Five-object vector collected from five-object questions

`run_bbh_five_mean.py` waits for the entire active reasoning-transfer queue
(including SVAMP and GSM8K) to exit, then acquires the shared GPU lock. It reuses
the existing five-object audio, prompt, and saved neutral/angry answers. A fixed
seed-42 shuffle assigns 125 questions to matched-forward mean collection and
125 different questions to L31 answer x1 evaluation. This is a separate output
folder; the previous three-object-vector transfer experiment remains intact.
No new TTS is needed. Since earlier five-object results were inspected, this
follow-up is exploratory.

```bash
nohup .venv-diffa/bin/python -u run_bbh_five_mean.py \
  > outputs/bbh-five-mean.log 2>&1 < /dev/null &
tail -f outputs/bbh-five-mean.log
.venv-diffa/bin/python run_bbh_five_mean.py --mode summary
```

New results, collection pairs, split/configuration and `style-vectors.pt` are
saved under `outputs/bbh-five250-angry-mean`. The installed upstream five-object
BBH task scores both strict and flexible answers. Collection uses up to 250
generations and evaluation up to 125 steered generations, plus any missing
neutral/angry test answers. Restarting resumes completed work.

## GSM8K strength sensitivity sweep

`run_gsm_strength_sweep.py` waits for active reasoning-transfer and mean-steering
follow-ups in this repository to exit, then takes the shared GPU lock. Run this
on the GSM8K/steering machine after launching the earlier follow-ups. Waiting
observes processes on that machine only; it does not coordinate a second machine.

The sweep uses the aligned `####` prompt, the same 718 evaluation questions and
original GSM8K vector, L31 answer tokens, seed 42 and unchanged decoding. Strengths
are -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5 and 2. Compatible neutral (strength 0)
and strength +1 responses are copied from `outputs/reasoning-transfer/gsm8k`.
With both complete, seven additional strengths require 5,026 generations.
No audio generation or vector recollection is needed.

```bash
nohup .venv-diffa/bin/python -u run_gsm_strength_sweep.py \
  > outputs/gsm-strength-sweep.log 2>&1 < /dev/null &
tail -f outputs/gsm-strength-sweep.log
.venv-diffa/bin/python run_gsm_strength_sweep.py --mode summary
```

Outputs are separate under `outputs/gsm8k-L31-strength-sweep`. Each strength is
compared against the same neutral baseline using upstream strict and flexible
scoring, additional numeric equivalence, paired confidence intervals and exact
McNemar tests. These are exploratory, unadjusted sensitivity comparisons; a newly
selected best strength is not independent validation of a tuned method.
