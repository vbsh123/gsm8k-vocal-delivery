# DIFFA L28 probe on the machine hosting Phi

This runs DIFFA, with the already collected vanilla angry/random direction.
It does not apply DIFFA vectors to Phi. It reuses exactly the same894 neutral
test questions, source baselines, prompt,512-token budget and x+1 strength as
the existing L31 run; only the intervention layer changes to28. No recollection.

On the original DIFFA machine:

```bash
git pull --ff-only
bash run_diffa_layer_probe.sh --mode pack --layer 28
```

Download the small file printed by the command:
`outputs/diffa-layer-probes/gsm8k-angry-random-L28-answer-a1.zip`.
Upload it to `/workspace/` on the Phi machine. It contains the raw mean-vector
file, frozen probe configuration, source neutral answers, exact model revisions
and software receipt. It contains no audio or model weights.

On the Phi machine, where the Jenny audio is already restored:

```bash
git pull --ff-only
nohup bash run_diffa_layer_probe.sh --bundle /workspace/gsm8k-angry-random-L28-answer-a1.zip >> outputs/diffa-L28-angry-random.log 2>&1 </dev/null &
```

Uses the existing `.venv-diffa`, Whisper cache and pinned DIFFA source. Missing
LLaDA/DIFFA weights download at model load; the environment and Whisper do not
need to be downloaded again. No Phi weights or outputs are removed. The wrapper
disables optional Xet caching to avoid extra download-cache disk consumption.
Shared GPU work is allowed to finish before loading the model.

Before steering, the probe verifies all target audio hashes and checks source
torch/transformers/peft versions and pinned model revisions. It replays three
seed42-sampled neutral responses. If they differ from the source, it records
`baseline-replay.json` and stops before steering rather than silently treating
the source baselines as reproduced. Passing a three-question replay is a limited
reproducibility check, not proof of identical behavior on all inputs.

The same launch command safely resumes a stopped probe. Restore verifies bundled
files, preserves appended steering answers, and refuses different existing files.
It never overwrites the original L31 run or its collection data.

Check scores without loading the model:

```bash
bash run_diffa_layer_probe.sh --mode summary
```

Results live under
`outputs/diffa-layer-probes/gsm8k-angry-random-L28-answer-a1/`.
All paired tests are exploratory and unadjusted. Compare the finished L28 result
with L31 on these identical target IDs, not partial subsets of different size.
