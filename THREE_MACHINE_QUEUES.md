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
