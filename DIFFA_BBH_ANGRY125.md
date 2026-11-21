# BBH3/BBH5 angry-only agreement, new125/125 split

This experiment uses only BBH3 and BBH5. For each task, select125 collection
questions whose angry clip emotion2vec recognizes as angry, then evaluate on the
remaining125 questions. Neutral recognition is irrelevant; neutral audio remains
paired with every angry collection clip. There is no happy collection, new TTS,
classifier inference, strength sweep or transfer scheduled by this launcher.

The existing audit found213/250 recognized angry clips for BBH3 and190/250 for
BBH5, enough to select125 each. Its original collection halves had only102 and88,
so fulfilling125 requires a new split. The selection keeps eligible original
collection questions first, fills from eligible original test questions with
seed42, and puts every unused question in the new test half. No response accuracy
is used for selection. The original experiments remain saved.

```bash
git pull --ff-only
# CPU-only split/reference/evaluator preparation.
bash run_diffa_bbh_angry125.sh --mode prepare

nohup bash run_diffa_bbh_angry125.sh \
  > outputs/diffa-bbh-angry125.log 2>&1 </dev/null &
tail -F outputs/diffa-bbh-angry125.log
```

The runner waits for the shared DIFFA GPU lock and idle GPU, then reuses one model
for BBH3 and BBH5 sequentially. It collects raw angry-minus-neutral means with the
original prompts, seed42,512-token budget, fast-dllm128 steps/block32 and matched
forward averaging. It applies L31 answer +1 on the new125 neutral test inputs.
Compatible baseline answers are reused. The moved test questions can lack saved
baseline answers, so only those missing neutral baselines are generated; no
baseline from a different question is substituted. Previously seen question IDs
are recorded explicitly and the follow-up remains exploratory.

All configurations, eligible/selected question IDs, classifier receipts, aggregate
checkpoints, collection raw answers, vectors and evaluation outputs are under
`outputs/diffa-bbh-angry-only125/<task>`. Summaries include paired scoring and
matched mean/median tokens using the original upstream BBH scoring.

```bash
bash run_diffa_bbh_angry125.sh --mode summary
```

There are two reports per task:

1. New vector versus neutral on all125 new test questions.
2. New versus original vector (plus neutral) on the common held-out questions,
   excluding all questions used to collect either vector. Given the current
   labels, this should contain102 BBH3 and88 BBH5 questions when both runs finish.

Do not compare the old full125 aggregate against the new full125 aggregate as if
the questions were identical. The common-held-out report provides the direct
paired vector comparison. Its source/raw answers and scoring provenance are saved
under `primary/common-heldout/`.

```bash
.venv-diffa/bin/python pack_experiment_backup.py --only bbh-angry125 \
  --destination /workspace --samples 3 --sample-mb 100
```

Run only this first. Review the complete primary and common-held-out results
before deciding whether to expand angry collection/strengths/transfer or return
to happy-vector experiments.
