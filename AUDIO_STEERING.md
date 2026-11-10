# Audio extraction and steering

## Large matched-forward experiment

```bash
git pull --ff-only
nohup bash run_large_steering.sh > outputs/jenny-large.log 2>&1 < /dev/null &
tail -f outputs/jenny-large.log
```

See every stage's completed and partial results at any time, without loading a model:

```bash
.venv-diffa/bin/python run_large_steering.py --summary-only
```

The launcher waits for the shared GPU lock and for existing GPU workers to finish. It runs one model process at a time. Run the same command after an interruption to resume saved generations and extraction pairs. Keep only one copy of this launcher queued.

Before starting, it checks the required saved pairs/audio files and validates the temporal collection against the original checkpoint. Resume tolerates an unfinished final JSON write while retaining complete answers; malformed completed rows still raise an error. Baseline reuse requires the same recording and generation recipe, but does not require identical timing metadata. Different baseline answers stop the run for review.

All evaluations use the unchanged 460 second-set Jenny neutral recordings; extraction uses the same first-set 425 emotion2vec-agreed angry/neutral pairs as the original direction. Every new generation resets literal seed42. Existing checkpoints and result folders are preserved. Independent all-forward averaging and direct answer-loss training are excluded.

The sequential stages are:

1. Re-evaluate the original direction unchanged at L12 acoustic ×1 with literal seed42.
2. Verify that the observational diffusion clock leaves two zero-steered responses and their forward counts exactly unchanged.
3. Average cached first-set differences under two filters: all angry-correct pairs; and all pairs except neutral-correct/angry-wrong. Evaluate both independently at L11 answer ×1, L12 acoustic ×1, and L16 answer ×1.25.
4. Collect the three target representations once for both temporal methods. Fit separate directions by within-decoding-block unmasking phase and by global 20-forward windows. Evaluate each method at all three targets.
5. With the original unchanged checkpoint, compare always-on steering against early, middle, late, first-forward-only, and every-other-forward schedules at all three targets. Reuse always-on answers only when checkpoint identity, recording, and literal42 recipe agree; generate missing controls.

Here **early/middle/late** mean the committed fraction of the current 32-token decoding block is below one third, from one third to below two thirds, or at least two thirds. They reset at each decoding block. Some generations jump over a phase. **Periodic2** steers the first, third, fifth… forward within each decoding block. **First-forward-only** steers once in the entire generation. These are our operational definitions, not a claim that one schedule is established as best for math reasoning.

For temporal collection, keep only the shared forward prefix of each angry/neutral pair. Global windows use aligned forwards 1–20, 21–40, etc. For phase directions, select the two recordings' forwards in the same phase within that shared prefix, truncate to the smaller phase count, and average their differences. Average contributing questions equally. Save contributor counts per bin; empty or beyond-range bins fall back to the overall matched mean. Tail windows may have very few contributors.

`collect_temporal_vectors.py` saves only the three target layer/group summaries per pair, rather than full token activations. `steer.py` observes token IDs through embedding hooks to select phase/window/schedule without changing the decoder. `run_large_steering.py` prepares the queue and reports its stages. Shared evaluation and scoring remain in `evaluate_audio_steering.py` and `steer_audio.py`.

Results from this grid are exploratory: the layer/strength choices were previously explored on this evaluation set. A selected improvement still needs a separate validation cohort.

Read these files in order:

1. `extract_vectors.py`: load DIFFA-2 and generate an answer from audio.
2. `extract_audio_vectors.py`: select the angry/neutral pairs, collect activations, and fit directions.
3. `steer.py`: apply a direction to token positions, install/remove hooks, and report paired accuracy.
4. `steer_audio.py`: run the recordings and interventions listed in a JSON plan.

The shared code uses the same prompt, token boundaries, generation settings, and seed42 recipe as the newer Jenny experiments. It does not modify saved results.

## Extraction

The defaults select the same first-set, emotion2vec-agreed angry/neutral recordings. Every pair has the same question and voice. Each layer and token group gets its own direction.

```bash
# Original averaging: keep the shared number of forwards in each pair.
.venv-diffa/bin/python -u extract_audio_vectors.py \
  --averaging matched --methods mean --output outputs/vectors-matched

# New averaging: use every forward independently in each recording.
.venv-diffa/bin/python -u extract_audio_vectors.py \
  --averaging all --methods mean --output outputs/vectors-all-forwards
```

Both give equal weight to each question. `matched` averages angry-minus-neutral over the shared forwards. `all` subtracts the two full-trajectory means.

Choose `--methods mean svd1 svd_mean probe` to fit all four methods from one collection. SVD rank, head training, and direction normalization are unchanged. `--fit-only` uses the saved per-question means without loading DIFFA. Existing SVD/probe collections can be reused with `--averaging matched`.

Checkpoints are saved under **`OUTPUT/METHOD/style-vectors.pt`**, for example `outputs/vectors-all-forwards/mean/style-vectors.pt`. These are fixed-direction checkpoints, not per-forward steering checkpoints. Collection saves each pair separately for resuming.

## Steering

A plan contains `recordings` and `conditions`. Each recording has `clip_id`, `question_id`, `audio_path`, and `answer`. Conditions look like:

```json
{
  "conditions": [
    {"name": "baseline", "interventions": []},
    {"name": "L12-acoustic-a1", "interventions": [
      {"layer": 12, "group": "acoustic", "strength": 1}
    ]}
  ]
}
```

Multiple entries inside one `interventions` list form a simultaneous combination. Separate conditions run separately. Include baseline first. Groups are `acoustic`, `semantic`, `combined`, and `answer`.

You can use an existing Jenny `run_config.json` as the plan to keep its recordings, order, and conditions:

```bash
.venv-diffa/bin/python -u steer_audio.py \
  --plan outputs/steer-jenny-angry-complete/run_config.json \
  --vectors outputs/vectors-all-forwards/mean/style-vectors.pt \
  --output outputs/steering-all-forwards
```

Use a new output folder for a new experiment. Relative audio paths use `--audio-root`; the Jenny plans normally have absolute paths. `--prepare-only` writes the plan without loading the model.

```bash
.venv-diffa/bin/python steer_audio.py \
  --output outputs/steering-all-forwards --summary-only
```

## Existing experiments

`evaluate_audio_steering.py` selects the vector methods and conditions, reuses compatible baselines, and calls `steer_audio.py` directly. Generation and accuracy scoring happen in that shared runner. Each method runs in a separate process, sequentially. Existing run configurations and cached baseline rules stay compatible. `run_jenny_svd_probe.sh` launches the SVD/probe experiment; `run_jenny_all_forwards.sh` launches the all-forwards mean experiment.

Historical launchers are in [`archive/steering/`](archive/steering/README.md). Their selection rules and old seed policies are preserved. Run them as modules from the repository root, for example:

```bash
.venv-diffa/bin/python -m archive.steering.steer_jenny_recovery74_promoted --summary-only
```

`python steer.py` remains a compatibility entry for the historical CLI. New experiments use `steer_audio.py`, which resets seed42 before every generation.

`run_jenny_original_L12_seed42.sh` reruns the original +2.6-point checkpoint at L12 acoustic ×1 on the same460 neutral recordings. It reads the original vector path from the old complete-run configuration, leaves that checkpoint unchanged, and reuses compatible literal42 baselines. This isolates the evaluation seed policy; it does not re-extract the vector. Results are saved separately under `outputs/steer-jenny-original-vector-L12-seed42/original_mean`. The launcher uses the shared GPU lock. The evaluator also accepts `--vectors CHECKPOINT --methods LABEL --conditions L12-acoustic-a1` for a direct single-condition comparison.

## Outcome-filtered and answer-trained vectors

`extract_benefit_vectors.py` reads a completed first-set collection and averages only pairs where angry solved the question and neutral failed. It uses the saved seed42 responses and activations, without another model run. Run it once with `outputs/jenny-svd-probe` (matched forwards), and once with `outputs/jenny-all-forwards` (all forwards).

`train_answer_vectors.py` freezes DIFFA and independently trains a vector for L11 answer ×1, L12 acoustic ×1, and L16 answer ×1.25. It uses only the first-set neutral audio and reference answers. The target text is `Final answer: <reference>` plus EOS. Randomly masked target positions are predicted with same-position cross-entropy; this is a masked diffusion objective, not causal next-token loss or backpropagation through greedy generation. Only the vector is optimized, with L2 regularization and activation checkpointing. Defaults are two epochs, Adam learning rate0.001, L2 coefficient0.01, seed42. Progress is saved every25 updates and after each target.

For answer training, **matched/all describe the initialization** from the respective mean checkpoints. The answer loss and training data are identical. There is no min-forward truncation during training and no post-training norm rescaling. Only the three listed layer/group vectors are trained; other entries retain their initialization.

`run_task_vectors_after_current.sh` waits for `run_jenny_all_forwards.sh` (or an explicit launcher PID), tests the gradient path on two questions, prepares all four method/averaging variants, then evaluates each on the same460 second-set recordings. A lock prevents duplicate follow-up queues. Each GPU stage runs sequentially. Results are under `outputs/steer-jenny-task-vectors/{matched,all}/{benefit_mean,answer_trained}`. All first-set question IDs are excluded from evaluation, including training pairs that the benefit filter discarded.

```bash
nohup bash run_task_vectors_after_current.sh \
  > outputs/jenny-task-vectors-night.log 2>&1 < /dev/null &
tail -f outputs/jenny-task-vectors-night.log
```

If needed, pass the current **bash launcher PID**, not just one Python child, as the script's first argument. Pass0 only when the GPU is already free. If the new GPU training path fails its smoke, the queue still extracts and evaluates both benefit-filtered variants. A partially trained checkpoint is not evaluated. The old `evaluate_jenny_svd_probe.py` name is a thin compatibility entry so already-started launchers can finish after an update.

## Broader outcome filters

`run_jenny_outcome_filters.sh` reuses the same first-set activations with two additional filters:

| Method | Angry right, neutral wrong | Both right | Both wrong | Neutral right, angry wrong |
|---|---|---|---|---|
| `exclude_losses_mean` | Keep | Keep | Keep | Exclude |
| `angry_correct_mean` | Keep | Keep | Exclude | Exclude |

Both are extracted with matched and all-forward averaging, then tested on the same460 second-set neutral recordings at L11 answer ×1, L12 acoustic ×1, and L16 answer ×1.25. No gradient training or new activation collection is needed. The launcher waits on the existing task queue's lock before using the GPU. Saved vectors and evaluation results use separate `jenny-outcome-filters` folders.

```bash
nohup bash run_jenny_outcome_filters.sh \
  > outputs/jenny-outcome-filters.log 2>&1 < /dev/null &
```
