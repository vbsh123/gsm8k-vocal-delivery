# Remaining Azure evaluations and existing-dataset emotion audits

Run from the repository root after `git pull --ff-only`. Nothing here synthesizes
new Azure audio or requires Azure keys. Every GPU launcher waits for the shared
`outputs/jenny-task-vectors.lock` and for existing GPU processes to finish.
Processes exit between models; launchers do not hold two DIFFA instances.

## Azure recording cohorts

- Earlier cohort: `azure-jenny500-extra` plus `azure-jenny500-neutral`, questions
  50–549. Existing Kimi/Qwen results cover this cohort.
- Next cohort: `azure-jenny-next500`, questions 550–1049, including neutral,
  angry, happy, sad, terrified and shouting where generated.
- Newest cohort: `azure-jenny-remaining-test`, questions 1050–1318, generated in
  neutral, angry, happy and sad. There are 269 planned questions, not 200.
- The separate 200-question AudioStretchy experiment uses original Spoken GSM;
  it is not the newest Azure cohort.

Accuracy excludes clips longer than DIFFA's existing 30-second limit. Pairwise
tables intersect the available recordings for the same question and voice.
Emotion classifiers do not filter generation. Reports include unfiltered,
emotion2vec-only, SenseVoice-only, both-agree and at-least-one-agrees subsets
when the required classifier files exist. Extra styles without exact classifier
labels are reported unfiltered. Confidence intervals are paired bootstrap
percentile intervals for the accuracy difference, with 200,000 resamples;
McNemar p-values and intervals are unadjusted.

## 1. Finish DIFFA accuracy on both later Azure cohorts

On the DIFFA machine, using its existing `.venv-diffa`:

```bash
mkdir -p outputs
nohup python -u run_azure_followups.py --stage diffa \
  > outputs/azure-followup-diffa.log 2>&1 < /dev/null &
tail -f outputs/azure-followup-diffa.log
```

Saved answers are skipped. No older Ryan TTS results are required. The launcher
reports each cohort separately and both cohorts combined. It includes every
requested style present in the plans; absent styles have no rows to evaluate.

## 2. Classify both later Azure cohorts

Use the existing emotion environment, or run `bash setup_emotion.sh` once.
The classifier now resumes saved predictions instead of overwriting them.

```bash
nohup python -u run_azure_followups.py --stage classify \
  > outputs/azure-followup-classifiers.log 2>&1 < /dev/null &
```

For CPU classification while DIFFA uses the GPU, append `--device cpu`.
This is slower but does not wait for or use the GPU. It runs both emotion2vec
and SenseVoice, then prints the matched filtered accuracy tables.

## 3. Qwen and Kimi on the same two later cohorts

On the second machine, copy the two audio folders with their `plan.jsonl` and
`results-emotion2vec.jsonl` / `results-sensevoice.jsonl` files to the same paths.
Use the already working `qwen-omni` and `kimi-audio` conda environments. If setting
up a fresh machine, use `setup_qwen_omni.sh` and `setup_kimi_audio.sh` respectively.

```bash
nohup bash -c '
  python -u run_azure_followups.py --stage qwen &&
  python -u run_azure_followups.py --stage kimi
' > outputs/azure-followup-qwen-kimi.log 2>&1 < /dev/null &
tail -f outputs/azure-followup-qwen-kimi.log
```

This evaluates all eligible clips, rather than the three-question smoke. It
resumes saved answers and preserves the existing prompts and seed 42.

## 4. Fixed winning steering on 200 newest Azure neutral questions

This freezes the first 200 eligible unique neutral questions from the newest
Azure cohort, with no outcome or emotion-classifier selection. It tests the
original angry-minus-neutral vector at **L12 acoustic ×1**, plus a fresh seed-42
baseline on the exact same recordings. The model/vector prompts and generation
settings are unchanged. Extraction overlap is rejected by the existing runner.
No vector is re-extracted or selected using the new answers.

```bash
nohup python -u run_azure_followups.py --stage steering \
  > outputs/azure-followup-steering.log 2>&1 < /dev/null &
tail -f outputs/azure-followup-steering.log
```

The original checkpoint is found through
`outputs/steer-jenny-angry-complete/run_config.json`. If it was moved, pass
`--vectors /actual/path/to/the/original/style-vectors.pt` to the launcher.

```bash
.venv-diffa/bin/python steer_azure_holdout.py --summary-only
```

Results are in `outputs/steer-jenny-remaining200-L12`. This is 400 answers including
baseline, and is separate from the speed experiment. To inspect/freeze selection
without running inference, call `.venv-diffa/bin/python steer_azure_holdout.py
--prepare-only` with any required `--vectors` argument.

## Expanded third-batch steering promotion after the completed large sweep

`run_jenny_promoted_third.py` tests all four distinct configurations whose exact
gain exceeded two percentage points on the completed second-set 460 questions:

| Direction | L12 acoustic strength | Schedule | Second-set gain |
|---|---:|---|---:|
| Original angry-minus-neutral mean | 1 | Always | +2.61 pp |
| Mean from angry-correct training pairs | 1 | Always | +2.17 pp |
| Different directions in global 20-forward windows | 1 | Always | +2.39 pp |
| Original mean | 1 | Every 2 forwards | +2.17 pp |

The original confirmation and original always-on timing row are the same
configuration and are counted once. No L11/L16 row in the completed large sweep
reached two percentage points. A fifth, new configuration is original L12
acoustic x1 every 3 forwards. Periodic schedules restart in each 32-token
decoding block: periodic2 uses forwards 1/3/5/... and periodic3 uses 1/4/7/....

Run on the original steering machine, where the completed big-run configs and
all three vector checkpoints are present. All five interventions run separately
on exactly the same third-batch neutral inputs. The default is every eligible
neutral question in `azure-jenny-remaining-test` (up to 269), no correctness or
classifier filtering. This is six answers per question including one shared
baseline, minus compatible saved baselines or completed results. Pass
`--num-examples 200` to restrict to the first 200 instead.

The final stage tests periodic3 on all 460 old second-set questions; it reuses
their compatible baseline answers. No vectors are re-extracted. The four
promotions and the additional schedule are fixed before new-cohort evaluation.
Checkpoint paths come from each completed experiment's `run_config.json`, so
the mean/filter/window directions cannot accidentally be substituted for one
another. The old 460-question periodic3 result remains exploratory.

```bash
nohup .venv-diffa/bin/python -u run_jenny_promoted_third.py \
  > outputs/jenny-promoted-third.log 2>&1 < /dev/null &
tail -f outputs/jenny-promoted-third.log

.venv-diffa/bin/python run_jenny_promoted_third.py --summary-only
```

The launcher preflights all plans/checkpoints, then waits for the common GPU
lock. Results are under `outputs/steer-jenny-third-promoted`: `original`,
`angry_correct`, `forward20`, and a separate `second-set-periodic3` directory.

## Read completed Azure results anytime

These read saved answers and classifier files and do not allocate a GPU:

```bash
.venv-diffa/bin/python analyze_jenny_deliveries.py \
  --paths outputs/azure-jenny-next500 outputs/azure-jenny-remaining-test \
  --styles neutral angry happy sad terrified shouting --confidence-intervals
```

For Qwen/Kimi, run in their conda environment and add
`--results-file results-qwen-omni.jsonl` or `--results-file results-kimi-audio.jsonl`.
Include the earlier two Azure folders in `--paths` to combine all cohorts.

## Existing datasets: classify which emotions are already present

This is a descriptive audio audit only. It does not synthesize emotional speech,
solve benchmark questions, or compare answer accuracy. It runs both classifiers.

```bash
.venv-emotion/bin/pip install pyarrow==19.0.1
nohup bash run_dataset_emotions.sh cuda:0 spoken-gsm mmau mmau-pro speechr \
  > outputs/dataset-emotions.log 2>&1 < /dev/null &
tail -f outputs/dataset-emotions.log
```

Replace `cuda:0` with `cpu` to run without waiting for the GPU. Pass only the
dataset names you want to run now; the default is all four. SpeechR must already
be downloaded and its nested audio ZIPs extracted under `outputs/speechr`.
For example, start with the smaller/local sources:

```bash
bash run_dataset_emotions.sh cuda:0 spoken-gsm speechr
```

The official sources are:

- Spoken GSM: `mistralai/gsm8k_speech`, all original test recordings. Unlike the
  DIFFA export, this audit includes recordings longer than 30 seconds.
- MMAU: both `gamma-lab-umd/MMAU-test` and `gamma-lab-umd/MMAU-test-mini` (not only
  test-mini). Original test metadata references its approximately 10.6-GB TAR;
  test-mini embeds audio in an approximately 1.2-GB parquet.
- MMAU-Pro: `gamma-lab-umd/MMAU-Pro`, using `test.parquet` and its approximately
  47.5-GB `data.zip`. Multiple-audio questions are handled per referenced file.
- SpeechR: every local audio recording and delivery variant under the supplied
  folder, with its parent directory retained as the variant/category.

Allow roughly **60 GB of additional disk space** if both full MMAU and MMAU-Pro
are not cached. Run these on a machine with space available. Archives are read
directly, without extracting a second full copy. Already extracted MMAU/Pro data
can instead be used with `--audio-root` in the direct command below.

Recordings are deduplicated by source audio path, retaining all referring question
IDs. The reports count audio files, not questions, and break down source speech,
music, sound or mixed categories. Classifier labels on music and environmental
sound do not establish speaker emotion. Whole recordings now use one classifier
call by default. Explicit `--chunked` enables the older duration-weighted
30-second aggregation and keeps it separate. The models' outputs are predictions, not emotion
ground truth. Dataset revisions, settings and per-model results are saved for
resume. An optional `--limit` is a smoke limit, not the default.

The feature extractor needs at least 400 samples (25 ms at 16 kHz). In explicit
chunked mode only, a shorter final chunk moves backward into the preceding chunk so every
real sample is still used exactly once, with no padding and no chunk above 30s.
Entire recordings shorter than 400 samples are saved as `insufficient_audio`,
excluded from predicted-emotion percentages, and counted separately. Existing
completed predictions are preserved. Failed inference messages include the clip
ID, chunk index and sample count to identify any remaining incompatible inputs.

```bash
# Small direct smoke; omit --limit for the full dataset.
.venv-emotion/bin/python -u classify_dataset_emotions.py \
  --dataset spoken-gsm --model emotion2vec --limit 5 --device cpu

# Existing extracted MMAU-Pro directory containing data/:
.venv-emotion/bin/python -u classify_dataset_emotions.py \
  --dataset mmau-pro --audio-root /workspace/mmau-pro --model emotion2vec

# Saved distributions; neither downloads audio nor loads a classifier.
.venv-emotion/bin/python classify_dataset_emotions.py \
  --dataset mmau-pro --model emotion2vec --summary-only
```

Default whole-recording reports and JSONL predictions live in
`outputs/dataset-emotions-whole/{spoken-gsm,mmau,mmau-pro,speechr}`. Once both classifiers
have run, summaries also show agreement over their shared recordings. Use a new
output folder when changing an audit's source directory or aggregation recipe.

### Correction: whole recordings are now the default

The earlier audit added 30-second chunking as an unverified memory precaution;
that is not an established emotion2vec maximum duration. The default now passes
the complete waveform in one classifier call. Old chunked results stay under
`outputs/dataset-emotions` and are never overwritten or mixed with whole-recording
predictions for longer audio. Compatible old results with exactly one complete
chunk are copied automatically: those already used the entire recording, so
only the missing or previously split recordings need new inference. The source
revision is inherited from the old audit when available.

```bash
nohup env STREAM_AUDIO=1 bash run_dataset_emotions.sh \
  cuda:0 spoken-gsm mmau mmau-pro speechr \
  > outputs/dataset-emotions-whole.log 2>&1 < /dev/null &
```

Memory failures are saved and counted separately, without automatic chunking.
To retry such failures with host memory, use the direct classifier command with
`--device cpu --retry-memory-failures`. Explicit `--chunked` restores the earlier
aggregation in its separate old output folder; it is not the default.
