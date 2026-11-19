# Three-worker SVAMP layer sweep

Run after the current experiments finish and room is available for three Phi models.
This launcher does not wait for a GPU queue and does not stop existing processes.
Each process checks available memory before loading; GPU execution is not tested locally.

```bash
git pull
nohup bash run_phi4_svamp_layers_parallel.sh \
  > outputs/phi4-svamp-layers-nightly.log 2>&1 </dev/null &
```

Defaults: existing 300 collection /300 disjoint test questions, answer-position
layers 0..31 except 17,28,29,30; strengths -1,+1; three workers. L0 answer
is distinct from the separate L0 audio experiment. This is an exploratory sweep
on previously inspected questions, not new confirmatory validation.

The assignments are round-robin (10/9/9 layers), frozen in
`outputs/phi4-sweeps/svamp300-300/nightly-assignment.json`.
All three subprocesses are launched before waiting for their completion. Each
loads one model, observes its layers together on each collection delivery, then
runs its layers/strengths sequentially using that same model. No layer/strength
setting is assigned to more than one worker. The same collection questions are
used by all workers; this intentionally duplicates unsteered collection forwards
three times to avoid rerunning collection separately for every layer.

Existing compatible collection pairs, vectors, test baselines and completed
answers are reused. Interrupted runs resume with the identical command. There
are 16,800 planned steered answers and at most 1,800 collection generations
before reuse. Baseline generation may be duplicated between workers if absent.
Per-setting and collection-file locks protect against duplicate writers; there
is no shared GPU lock. Do not run an overlapping sweep on these same settings.

```bash
tail -F outputs/phi4-svamp-layers-nightly.log \
  outputs/phi4-sweeps/svamp300-300/nightly-worker0.log \
  outputs/phi4-sweeps/svamp300-300/nightly-worker1.log \
  outputs/phi4-sweeps/svamp300-300/nightly-worker2.log
```

To report the default settings:

```bash
bash run_phi4_sweep.sh --mode summary \
  --datasets svamp \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 19 20 21 22 23 24 25 26 27 31 \
  --strengths -1 1 --collection-size 300 --test-size 300 \
  --output outputs/phi4-sweeps/svamp300-300
```

Override `--layers`, `--strengths`, or `--workers` on the launcher if needed.
Changing collection/test sizes requires a separate `--output` directory.
