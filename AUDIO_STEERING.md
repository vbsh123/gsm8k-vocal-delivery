# Audio extraction and steering

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
