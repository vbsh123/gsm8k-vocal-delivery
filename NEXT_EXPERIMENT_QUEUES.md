# Two-machine follow-ups

Pull main on both machines. Azure speech credentials are required only where audio is missing. No existing runs are stopped or caches deleted.

## Phi machine

Collection is complete and the all-layer test phase has not started. Run:

```bash
nohup bash run_phi4_layers_then_prepare.sh > outputs/phi4-layers-then-prepare.log 2>&1 </dev/null &
tail -F outputs/phi4-layers-then-prepare.log outputs/phi4-all-layers-steer-worker0.log outputs/phi4-all-layers-steer-worker1.log
```

The existing two-worker test completes missing neutral/angry references on the 869 noncollection questions, freezes neutral-wrong/angry-correct cases, and tests all 32 layers at answer/audio positions at −1. These outcome-selected percentages diagnose recovery; they are not full-cohort accuracy estimates. The runner has no shared GPU queue or job dependency. Worker memory limits and duplicate-writer guards remain.

After both workers finish, CPU/TTS preparation freezes SVAMP 500 collection /500 test and BBH3/BBH5 125 collection /125 test. It reuses compatible Jenny audio and generates missing clips. No new task model inference runs yet. The exact prompt and test/collection IDs are saved in each data_config.json. Each later task uses its own raw angry-minus-neutral mean vector.

Inspect the diagnostic with:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python phi4_all_layer_diagnostic.py --mode summary
```

After choosing the layer, run all three tasks with an explicit choice (replace N and S):

```bash
nohup bash run_phi4_prepared_tasks.sh --layer N --group answer --strength S > outputs/phi4-prepared-tasks.log 2>&1 </dev/null &
```

Each task collects on its first half and evaluates the entire other half in neutral, angry and steered-neutral conditions. The shell releases each model process before the next task. There is no GPU queue/wait. BBH uses installed upstream task scoring. SVAMP uses installed GSM8K answer filters/metric on SVAMP references plus numeric equivalence and the authors' numeric tolerance; it is not an official SVAMP inference task. All are spoken adaptations.

## DIFFA machine

Transfer was not previously launched. Run:

```bash
nohup bash run_diffa_voice_then_transfer.sh > outputs/diffa-voice-transfer.log 2>&1 </dev/null &
tail -F outputs/diffa-voice-transfer.log outputs/multiarith-guy.log outputs/winning-transfer.log
```

First generate/resume neutral/angry Azure en-US-GuyNeural audio on the same 290 MultiArith evaluation questions. Same original prompt, seed42 and native DIFFA generation recipe; no collection/steering. Evaluation waits for the existing shared GPU lock and active GPU work. CPU scoring follows. Then the existing transfer runner applies every original task vector to every other task at L31 answer ×1: 30 off-diagonal pairs across GSM8K, SVAMP, MultiArith, IFEval, BBH3 and BBH5. Transfer also waits for GPU access. Target prompts, cohorts and budgets stay fixed.

## Evidence ZIPs tomorrow

Phi:

```bash
.venv-diffa/bin/python pack_experiment_backup.py --only phi --destination /workspace --samples 3 --sample-mb 100
```

DIFFA:

```bash
.venv-diffa/bin/python pack_experiment_backup.py --only strengths voice transfer --destination /workspace --samples 3 --sample-mb 100
```

Download the printed ZIP and SHA256 file. Packages include raw textual responses, configs/splits, vectors, code, scorer provenance and matched sample audio where available. Phi includes small collection caches with raw collection responses and means. The manifest reports saved/planned counts and missing artifacts. Packaging a running transfer queue captures completed answers so far; it does not make unfinished experiments complete. Full model weights and full audio banks are excluded.

## Sequential Phi nightly continuation (L17 answer −1.5)

Once the original diagnostic launcher is already running, queue:

```bash
nohup bash run_phi4_nightly_followups.sh > outputs/phi4-nightly-followups.log 2>&1 </dev/null &
tail -F outputs/phi4-nightly-followups.log outputs/phi4-opposite-recovery.log outputs/phi4-prepared-tasks-L17-minus1.5.log
```

This supersedes the earlier manual choose-layer step for this requested nightly experiment: L17 answer −1.5 is explicit. It does not stop or relaunch the current diagnostic. It waits for that diagnostic and its preparation wrapper to exit, verifies all diagnostic answers are complete, then takes the shared GPU lock and waits for active GPU work to end. All added model runs are sequential.

First the opposite diagnostic selects neutral-numeric-correct / angry-numeric-wrong among the same 869 noncollection questions, using the original fixed all-layer400 vectors. All32 layers, answer/audio groups, at −1 are tested independently on both angry input (recovery) and neutral input (correctness retention). This remains an outcome-selected diagnostic without significance tests. It preserves the existing neutral-wrong/angry-correct diagnostic intact. No extra vector collection is needed for this GSM diagnostic.

Then SVAMP500/500 and BBH3/BBH5 125/125 are prepared/resumed and run at L17 answer −1.5, each collecting its own angry-minus-neutral vector and evaluating all neutral, angry and steered-neutral test questions. This part is full-cohort held-out evaluation, with ordinary unadjusted paired reporting. Each task releases its model before the next starts. The new opposite results are included in `pack_experiment_backup.py --only phi`.

## Focus the opposite diagnostic on shortlisted answer layers

```bash
git pull --ff-only
.venv-diffa/bin/python stop_phi4_nightly_opposite.py
nohup bash run_phi4_nightly_followups.sh --focused > outputs/phi4-nightly-focused.log 2>&1 </dev/null &
tail -F outputs/phi4-nightly-focused.log outputs/phi4-opposite-recovery.log outputs/phi4-prepared-tasks-L17-minus1.5.log
```

The helper stops only this repository's nightly wrapper/opposite diagnostic. The focused continuation runs L2/L6/L7/L17/L30 answer at minus1 on both input styles, skipping completed answers. It preserves the frozen original config and all prior results, including audio targets, and records the new execution scope separately. The summary still displays every saved target. The queue then continues to SVAMP/BBH3/BBH5 at L17 answer minus1.5. It does not automatically choose a different layer or launch full869 GSM evaluation.

## Guy happy/sad extension

```bash
git pull --ff-only
nohup bash run_multiarith_guy_happy_sad.sh > outputs/multiarith-guy-happy-sad.log 2>&1 </dev/null &
tail -F outputs/multiarith-guy-happy-sad.log
```

The DIFFA machine generates only happy/sad Guy clips on the same290 MultiArith questions. It copies all580 completed neutral/angry textual answers into a separate four-emotion output, retaining source hashes and leaving original files untouched. Same original prompt, seed42, fast_dllm128steps/block32/512tokens, temperature0, no steering or vector collection. Only580 new model calls are needed. TTS can run alongside transfer; native evaluation waits on the existing shared GPU lock, so a running transfer queue finishes first. No transfer process is stopped or relaunched. The four-emotion report compares each emotion against neutral using installed GSM8K filters/metric plus numeric equivalence. These additional comparisons were chosen after observing the first two conditions and are exploratory, with unadjusted tests. The `voice` backup group includes the added raw answers/configs/audio samples.

CPU summary:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python run_multiarith_second_voice.py --mode summary --output outputs/multiarith290-guy-four-emotions
```

## Preserve both emotion classifiers on all surviving audio

```bash
git pull --ff-only
nohup bash run_available_audio_emotions.sh > outputs/available-audio-emotions.log 2>&1 </dev/null &
tail -F outputs/available-audio-emotions.log
```

This waits only for active TTS preparation, then inventories all surviving audio under outputs and resolvable experiment references. Byte-identical recordings are classified once, with every source/voice/requested-style alias retained. emotion2vec and SenseVoice run sequentially on GPU, alongside transfer if VRAM permits; CPU is an optional first argument. Whole recordings are resampled to mono16kHz, without cropping or a silent chunk fallback. Errors/OOMs are saved explicitly and appear in coverage summaries. No answer or emotion outcome filtering is applied.

Audit files live in outputs/available-audio-emotions: audio_inventory.jsonl, inventory_manifest.json, both results-MODEL.jsonl files, runtime receipts and summary.json. Each prediction has the audio SHA256. emotion2vec retains its full class scores; SenseVoice retains its emotion tag and raw tagged transcript (not invented probabilities). Identical-byte aliases can include multiple requested styles; summaries count one hash per voice/style. Classifier labels are not human ground truth.

The audit does not regenerate deleted audio. Missing, ambiguous or mismatched references are listed. Old saved classifier files are retained in the final ZIP separately; they are not silently mixed into this new hash-verified audit. Audio generated after inventory will require a later inventory/resume. Completed hashes are skipped; `--retry-errors` on classify explicitly retries failed hashes, with the latest attempt used by summaries.

After both models finish, the wrapper creates the evidence ZIP automatically. To make another snapshot manually:

```bash
.venv-diffa/bin/python pack_experiment_backup.py --only strengths voice transfer emotions --destination /workspace --samples 3 --sample-mb 100
```

This includes all current raw strength/transfer answers, configs, vectors, the complete classifier audits, historical label files recorded in the inventory manifest, code and up to100MiB sample audio. Running experiments are partial snapshots with counts, not implicitly completed runs. Full model weights and all waveform banks are excluded. Keep the printed ZIP and SHA256 file.

## Replace the L17 task queue with full GSM L7 evaluation

```bash
git pull --ff-only
.venv-diffa/bin/python stop_phi4_nightly_opposite.py --include-l17-tasks
nohup bash run_phi4_gsm_L7.sh > outputs/phi4-gsm-L7-minus1-minus1.5.log 2>&1 </dev/null &
tail -F outputs/phi4-gsm-L7-minus1-minus1.5.log
```

This cancels the superseded nightly/opposite chain and, if already started, only its L17 answer minus1.5 task wrapper/native model in this repository. Completed files are retained. Existing CPU audio preparation may finish; no new L17 SVAMP/BBH model runs are scheduled. The new runner waits for GPU access and evaluates all869 noncollection GSM questions at L7 answer minus1 and minus1.5. It reuses the original400-question L7 vector, all869 neutral baselines, and matching L7 minus1 neutral-input diagnostic answers. It neither recollects vectors nor regenerates audio nor launches SVAMP/BBH. The869 are the same full comparison cohort used for L17/L30, rather than truncating to800. There are1738 planned steered answers minus reused matching diagnostic answers; the869 neutral responses are copied with provenance.

Layer7 was selected using subsets of these test questions, so this is an exploratory full-cohort follow-up, not an untouched confirmatory layer comparison. The exact prior inputs, source vector element and reuse provenance are checked and recorded. Saved output is included in the Phi evidence ZIP.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python run_phi4_gsm_L7.py --mode summary
```
