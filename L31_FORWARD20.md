# Queued L31 forward-window experiment

This follows the current `finish_jenny_finalists.py` queue. It waits for the
parent queue and its evaluation workers to exit, then waits for the common GPU
lock and any remaining GPU processes. Only one DIFFA process runs at a time.
This pipeline has its own lock to reject accidental duplicate launches.

On the steering machine:

```bash
git pull --ff-only
nohup .venv-diffa/bin/python -u run_L31_forward20_after_current.py \
  > outputs/jenny-L31-forward20.log 2>&1 < /dev/null &
tail -f outputs/jenny-L31-forward20.log
```

No Azure credentials or new audio generation are needed. CPU-only preparation
checks the existing inputs before waiting. Use `--prepare-only` to stop there.

## Stages

1. Collect only **L31 answer** states using the original 425 first-cohort
   angry/neutral pairs. Each recording resets seed42. Match forwards up to the
   shorter trajectory, average answer positions per forward, then average the
   angry-minus-neutral differences in global windows 1–20, 21–40, etc. Each
   contributing question has equal weight in its window. The checkpoint also
   retains an overall matched mean, used for absent or beyond-range windows.
   Sparse late windows report their contributor counts.
2. Evaluate this new direction at **L31 answer ×1, always-on**, on the same 460
   second-cohort neutral recordings and 258 eligible final-cohort neutrals.
3. Finish the existing original-mean **L31 answer ×1** and **L31 answer ×1
   periodic2** conditions on the final 258 recordings. Reuse compatible baselines
   and any already completed selected answers. The other L29/L30 conditions do
   not run, even though their old frozen config is retained.

There are up to 850 collection generations, 718 new window-steering generations,
and 516 original-mean final-cohort generations, reduced by saved work. Baselines
are reused when recording and recipe match. Directions are learned only from
the first cohort, not from either evaluation cohort. Original and window vectors
are different methods and are reported separately.

## Outputs and progress

```bash
.venv-diffa/bin/python run_L31_forward20_after_current.py --summary-only
cat outputs/jenny-L31-forward20/status.json
```

The summary prints combined results across completed pairs from both evaluation
cohorts. During a run, pair counts reflect available answers, not the final 718.

- `outputs/jenny-L31-forward20/vectors/pairs/`: resumable per-question differences.
- `outputs/jenny-L31-forward20/vectors/forward_windows/`: the L31 checkpoint and
  contributor/extraction metadata.
- `outputs/jenny-L31-forward20/evaluation/second/forward20/`: 460-question evaluation.
- `outputs/jenny-L31-forward20/evaluation/third/forward20/`: 258-question evaluation.
- `outputs/steer-jenny-late-layers/third-cohort/original_mean/`: the two selected
  original-vector L31 conditions, stored with the existing cohort.
- `outputs/jenny-L31-forward20/combined-summary.json`: final pooled statistics.

Repeat the same launch command after an interruption to resume saved pairs and
answers. A failed stage stops the queue and records its name and return code.
Collection and each evaluation run in separate subprocesses so model memory is
released between stages. The previous L12 checkpoint and evaluations are untouched.

To inspect the current 460-question L31 finalists, independently of this queue:

```bash
.venv-diffa/bin/python steer_audio.py \
  --output outputs/steer-jenny-late-layers/original_mean --summary-only \
  --conditions L31-answer-a1 L31-answer-a1-periodic2
```
