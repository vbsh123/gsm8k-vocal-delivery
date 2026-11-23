# Three-machine experiment queues

The queues preserve the existing original experiments and write all new
experiments into separate directories. All generation is seeded and uses the
existing model-native audio paths. No command below regenerates TTS or runs
emotion2vec inference. The audio ZIP and saved exact-hash labels are reused.

## Original DIFFA machine

The original happy queue uses exactly the contributing angry collection IDs
(GSM425, not the larger500 exclusions), the original held-out questions, and
the original task-specific collection/evaluation prompts and budgets. It
collects happy-minus-neutral and steers at zero-based L31 answer +1. Happy is
not classifier-filtered. It also completes the unsteered happy comparison;
existing compatible answers and complete vectors are reused.

First make the shared-GSM handoff after BOTH collections finish:

```bash
git pull --ff-only
nohup bash run_three_machine_handoff.sh --wait --handoff > outputs/three-machine-handoff.log 2>&1 </dev/null &
```

`--handoff` waits for verified complete angry/happy vectors and collection
receipts, then sends SIGTERM only to the worker running
`diffa_gsm_shared_emotions.py` for the default shared-GSM output. A current
unfinished steering response may be repeated on the new machine, while saved
responses and complete vectors are retained. It does not stop any original
happy, Phi, or unrelated job. This avoids paying to finish the same shared
happy steering on both GPUs. Omit `--handoff` to leave the worker running.

The packager prints a ZIP path and checksum. Transfer these to `/workspace`
on the NEW machine alongside the audio ZIP. It includes only completed NEW
agreed vectors (not original-method vectors), the frozen configs, original
source templates/IDs/reference metadata, all saved e2v results and classifier
receipts, and raw results including shared-GSM angry. No weights/audio or
unfinished activation checkpoints are included.

The handoff also records the source cache's exact LLaDA, Whisper and DIFFA model
snapshot revisions. Both the original machine's next model load and the new
machine use that receipt; model weights are downloaded on the new machine,
without following a potentially newer floating `main` revision.

Queue the original happy work; it waits for the shared GPU queue:

```bash
nohup bash run_diffa_original_happy_queue.sh > outputs/diffa-original-happy.log 2>&1 </dev/null &
```

Completed original GSM/SVAMP work is reused. `--require-existing-audio` prevents
unexpected Azure calls. Do not run a second original happy queue concurrently.

Summary:

```bash
bash run_diffa_happy_nightly.sh --mode summary
python3 report_three_machine_queues.py
```

## Original DIFFA sad (queued after happy)

Sad uses the SAME ordered original happy/angry collection and held-out IDs,
neutral audio hashes, collection/evaluation prompts and budgets. It requires the
existing Jenny sad audio and never classifies, filters by e2v, substitutes a
question or requests new TTS. It collects sad-minus-neutral across layers,
reports collection and held-out sad versus neutral separately, then steers
held-out neutrals at L31 answer +1. Compatible neutral answers are reused.

Queue it on the ORIGINAL DIFFA machine while happy is still running:

```bash
git pull --ff-only
nohup bash run_diffa_original_sad_queue.sh > outputs/diffa-original-sad.log 2>&1 </dev/null &
```

CPU preparation validates all six cohorts/audio/reference metadata first. The
process then blocks on the SAME `outputs/jenny-task-vectors.lock` held by happy,
and waits for active GPU processes before loading weights. It makes no competing
model allocation while waiting. Completed work resumes in
`outputs/diffa-sad-winners`. No strengths grid or transfers are started.

```bash
tail -n 30 outputs/diffa-original-sad.log
bash run_diffa_original_sad_queue.sh --mode summary
python3 report_three_machine_queues.py
```

## Phi machine

Fresh ZIP experiment: angry-minus-neutral at L0 audio, L11 answer and L30 answer; each steered separately at strength -1.
Phi must use the EXACT original DIFFA collection AND evaluation IDs across all
six datasets, including GSM425/718. Do not use the old Phi400/869 cohort.
DIFFA's frozen configs supply the ordered IDs, exact neutral audio hashes,
question/reference metadata, collection/evaluation prompts and token budgets.
This is the original-protocol DIFFA cohort, not the separate shared-three study.

On ORIGINAL DIFFA, export the small cross-model cohort file (no inference):

```bash
git pull --ff-only
.venv-diffa/bin/python export_diffa_cohorts.py
```

Copy `/workspace/diffa-original-cohorts.json` to `/workspace` on Phi. The existing
audio ZIP already supplies the recordings; no audio re-upload is needed.

On PHI, after staging the Jenny ZIP:

```bash
git pull --ff-only
bash run_phi4_zip_recheck.sh --mode prepare
nohup bash run_phi4_zip_recheck.sh > outputs/phi4-zip-recheck.log 2>&1 </dev/null &
```

Preparation prints the exact DIFFA collection/evaluation counts and refuses a
missing or incompatible cohort file. No old Phi split, audio, vectors or answers
are reused. First collect fresh neutral/angry vectors and evaluate deliveries
on both collection and evaluation IDs across all datasets. All three directions
are captured during the SAME neutral/angry generation forwards. Once all six
datasets finish collection and delivery evaluation, release the collection model
and launch THREE concurrent steering workers. One worker runs L0 audio -1 across
all datasets, one L11 answer -1, and one L30 answer -1. Each uses the exact held-out IDs. Collection and evaluation each preserve DIFFA's
respective prompts and budgets. Reports include paired scores and token lengths.

Results: `outputs/phi4-diffa-cohorts-three-directions`. Per-direction progress
is in `steering-L0-audio.log`, `steering-L11-answer.log`, and
`steering-L30-answer.log` inside that directory. Per-task summaries are
`summary-L0-audio.json`, `summary-L11-answer.json`, `summary-L30-answer.json`;
unsteered delivery summaries are `delivery-summary.json`. The earlier attempted runs stay
separate and are not treated as matching-cohort comparisons. This remains
exploratory work on previously inspected questions and selected layers.

```bash
bash run_phi4_zip_recheck.sh --mode summary
python3 report_three_machine_queues.py
```

## Phi GSM older-prompt ablation

This separate run evaluates only the SAME718 held-out GSM questions with the
SAME fresh audio, model revision, seed and token budget. It changes only
"Give concise step-by-step reasoning with only the necessary calculations."
to the older "Let's think step by step." Both neutral and angry are generated;
the425 collection questions remain excluded. No vectors or steering are run.
The main queue and its artifacts are unchanged. This concurrent run checks
available GPU memory immediately and does not wait on the main GPU queue.

```bash
git pull --ff-only
nohup bash run_phi4_gsm_old_prompt.sh > outputs/phi4-gsm-old-prompt.log 2>&1 </dev/null &
bash run_phi4_gsm_old_prompt.sh --mode summary
```

Saved in `outputs/phi4-gsm-fresh-audio-old-prompt`. The summary reports the
older-prompt angry-versus-neutral effect; `prompt-comparison.json` compares
both prompt conditions on common completed question/voice IDs, requiring exact
audio hashes and budgets. These remain exploratory prompt ablations.

## New DIFFA A40

Upload/Cloud Copy both COMPLETE ZIPs and checksum files into `/workspace`.
Clone into the original absolute repo path, then install/restore:

```bash
cd /workspace
git clone https://github.com/vbsh123/gsm8k-vocal-delivery.git
cd /workspace/gsm8k-vocal-delivery
bash setup_new_diffa_machine.sh
```

If already cloned, use `git pull --ff-only`. If setup was already completed
before the new vector bundle arrived, restore just the handoff:

```bash
python3 restore_diffa_bundle.py /workspace/diffa-agreed-runs-*.zip
.venv-diffa/bin/python check_diffa_migration.py
```

Check environment differences, then replay ten saved neutral cases before
reusing old-machine baselines:

```bash
bash run_diffa_replay.sh --run
```

When that passes, prepare and launch:

```bash
bash run_diffa_shared_three_queue.sh --mode prepare
nohup bash run_diffa_shared_three_queue.sh > outputs/diffa-shared-three-queue.log 2>&1 </dev/null &
```

GSM uses the already frozen609/660 shared split, reuses both completed vectors
and saved angry results, and finishes the missing happy steering. It never
recollects GSM and does not change its cohort because new audio was added later.
For the other five datasets, collect angry and happy from the SAME shared-three
IDs, with the collection capped at half the available neutral pool. All three
emotional clips must be classifier-recognized and match the neutral's exact
question text/reference; neutral recognition is not required. The remainder is
evaluated without outcome/emotion filtering. Both collection and evaluation use
the same task-specific zero-shot target prompt. Steering is L31 answer +1.
No sad vector, transfer grid or additional strengths run automatically.

The new counts are determined from the latest existing audio/labels. Newly
generated SVAMP angry recordings may increase that task's eligible shared pool.
Each task writes its selected IDs into a frozen `split.json`; it never resplits
an existing prepared output.

Reports include each vector versus neutral AND a direct question-paired happy
versus angry steering comparison. Different significance labels alone do not
establish a difference between emotions.

```bash
bash run_diffa_shared_three_queue.sh --mode summary
python3 report_three_machine_queues.py
```

## End-of-run reports

On each machine, refresh its queue summary and run
`python3 report_three_machine_queues.py`. This produces
`outputs/three-machine-report/summary.json` and `summary.csv`. Download and name
the three JSONs separately, then combine them on one machine:

```bash
python3 report_three_machine_queues.py --inputs original.json phi.json shared.json
```

The merger prefers more complete snapshots of the same frozen run and rejects
conflicting runs/statistics instead of choosing larger gains. Primary is
numeric equivalence for math, flexible extraction for BBH and strict prompt
compliance for IFEval; strict and flexible metrics are also retained. Missing
remote summaries are shown as unavailable, never as failed experiments. All
tests/intervals remain exploratory and unadjusted.

## New DIFFA GSM prompt × vector ablation (original and shared3)

Use the completed ORIGINAL angry/happy vectors on425/718 and the completed
shared3 angry/happy vectors on609/660. Vectors remain fixed. Each protocol runs
its own frozen neutral-audio evaluation questions with BOTH the current concise
prompt and the older Phi "Let's think step by step" prompt, keeping `####`,
model, seed, budget, and L31 answer +1 fixed. Conditions per prompt are neutral,
angry-vector steering and happy-vector steering. No emotional-audio delivery,
TTS, e2v inference or vector collection is scheduled.

The previous handoff intentionally omitted original-method vectors. On ORIGINAL
DIFFA, export only the missing original GSM vectors, configs and saved answers:

```bash
git pull --ff-only
.venv-diffa/bin/python pack_diffa_gsm_prompt_inputs.py
```

Download the printed `/workspace/diffa-gsm-original-prompt-inputs-*.zip` and its
`.sha256`, upload both to `/workspace` on NEW DIFFA, then:

```bash
git pull --ff-only
.venv-diffa/bin/python restore_diffa_bundle.py /workspace/diffa-gsm-original-prompt-inputs-*.zip
bash run_diffa_gsm_prompt_ablation.sh --mode prepare
nohup bash run_diffa_gsm_prompt_ablation.sh > outputs/diffa-gsm-prompt-ablation.log 2>&1 </dev/null &
```

The runner waits on the current shared DIFFA GPU lock before model loading.
Preparation reuses compatible concise-prompt answers and reports missing counts.
After waiting it refreshes that reuse, so any newly completed main-queue answers
are used too. Prompt-matched neutral answers are reused across protocol cohorts
where their question IDs/audio hashes/recipes agree. Baselines from the concise
prompt are never reused for the older prompt.

Each protocol has up to six conditions per question, with compatible existing
answers reused. Results are in `outputs/diffa-gsm-prompt-ablation/{original,shared3}`.
Reports provide gains against each prompt's own baseline, direct paired
happy-versus-angry tests and older-versus-concise prompt comparisons. The root
`common-heldout-comparison.json` compares original/shared3 vectors ONLY on IDs
held out from BOTH vector collections, with identical audio/prompt/budget checks.
Original718 and shared3 660 headline scores are not treated as the same cohort.
All comparisons remain exploratory and unadjusted.

```bash
tail -n 30 outputs/diffa-gsm-prompt-ablation.log
bash run_diffa_gsm_prompt_ablation.sh --mode summary
```
