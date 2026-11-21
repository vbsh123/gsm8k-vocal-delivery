# Classifier-agreed happy GSM vector

This separate experiment collects happy-minus-neutral directions only where
emotion2vec recognizes both neutral and happy correctly. Labels are joined to
the exact saved audio, with audio hashes and classifier-file receipts preserved.
No model-correctness filtering is used.

The archived GSM labels have 263 agreed pairs within the original 425 angry-selected
questions, or 283 within the full original 500-question excluded collection pool.
Neither pool contains 425 happy-agreed pairs. The preflight reports the exact count
with audio available on the execution machine; it never fills with unrecognized
clips or borrows from the test split.

Choose `--pool existing425` or `--pool original500` explicitly. Each has a separate
output under `outputs/diffa-gsm-happy-agreed/<pool>`. The previous happy and angry
collections and results are preserved. This is the same filtering *rule* as angry,
not an equal-sized vector comparison unless contributing pairs are additionally
matched between emotions.

The original 718 held-out neutral questions, answers, evaluation prompt (`####`),
512-token budget and decoding settings are preserved. Collection uses the saved
`Final answer:` prompt. The vector is applied at L31 answer positions, strength +1.
There are no new held-out neutral or emotional-delivery generations, other tasks,
transfers or strength sweeps. The existing happy vector cannot be subsetted from
its aggregate sums, so the agreed collection pairs are forwarded again, with
resumable atomic running checkpoints. Neutral collection forwards are necessary;
they are distinct from regenerating the held-out baseline.

Preparation is CPU-only, writes the selected configuration and prints its count:

```bash
bash run_diffa_happy_agreed.sh --pool existing425 --mode prepare
```

After choosing the pool, launch once. It waits for the shared DIFFA GPU lock and
idle GPU; it does not stop the existing happy queue:

```bash
nohup bash run_diffa_happy_agreed.sh --pool existing425 \
  > outputs/diffa-gsm-happy-agreed-existing425.log 2>&1 </dev/null &
bash run_diffa_happy_agreed.sh --pool existing425 --mode summary
```

Substitute `original500` consistently to use the larger original collection pool.
No Azure generation or classifier inference is launched by this script.
