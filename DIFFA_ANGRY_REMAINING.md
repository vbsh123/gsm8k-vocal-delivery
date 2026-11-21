# Remaining angry-only agreed collections

This runs SVAMP, MultiArith and IFEval sequentially with one DIFFA model, after
waiting for the shared GPU lock and idle GPU. It is separate from the BBH3/BBH5
125-pair experiment. Only the angry clip must be classified as angry by
emotion2vec; neutral recognition is not a selection requirement.

| Dataset | New collection | Remaining test | Original eligible collection clips |
|---|---|---|---|
| SVAMP | 500 | 500 | 493/500 |
| MultiArith | 290 | 290 | 272/290 |
| IFEval | 254 | 254 | 248/254 |

Keep eligible original collection questions first, then fill the half from
eligible original test questions using seed42. Evaluate every remaining question
without classifier filtering. The new split is exploratory and fixed before
generation. There is no selection by response correctness. Original experiments
remain saved. Original token budgets, prompts, matched-forward raw mean extraction,
seed42,128 steps/block32 and L31 answer+1 are preserved; IFEval uses1024 tokens,
math uses512. Compatible neutral baselines are reused; missing new-test baselines
are generated. Reports include a direct original/new comparison on questions
held out from both vectors, plus primary results on the whole new test half.

## Restore the existing SVAMP angry test audio from Phi

DIFFA's earlier audit had500 SVAMP angry recordings, all already classified;
493 were recognized. Its test-half angry recordings were absent on that machine,
not simply unclassified. Phi already evaluated those deliveries. Export its
existing500 test-half angry clips without TTS/inference:

```bash
# On Phi:
git pull --ff-only
.venv-diffa/bin/python diffa_angry_remaining.py --mode export-svamp
```

Download/copy `/workspace/svamp-phi-angry-test500.zip` to the DIFFA machine. Then:

```bash
# On DIFFA, assuming the ZIP was copied to /workspace:
git pull --ff-only
.venv-diffa/bin/python -m zipfile -e /workspace/svamp-phi-angry-test500.zip \
  outputs/imported-phi-svamp-angry-test500
```

The archive contains the original Jenny Azure clips, their audio plan and hash
receipts. Original DIFFA neutral audio is never replaced. This copy is needed to
classify the full angry pool and choose500 agreed clips without generating audio.

## Audit all available clips, then prepare and launch

```bash
bash run_diffa_angry_remaining.sh --mode audit
bash run_diffa_angry_remaining.sh --mode prepare
nohup bash run_diffa_angry_remaining.sh \
  > outputs/diffa-angry-remaining.log 2>&1 </dev/null &
```

Audit uses an isolated `outputs/diffa-angry-only-remaining/audit`, preserves the
BBH audit, reuses exact-hash whole-recording labels from previous audits, and
classifies all missing neutral/angry/happy labels in these three datasets. It
waits for GPU classifier work and does not collect vectors or generate answers.
Preparation validates full audio/angry-label coverage and sufficient recognized
clips before the queued DIFFA model startup. It never silently reduces collection
size or uses unrecognized clips to fill a half. No happy vectors, strengths or
transfer run automatically.

```bash
tail -F outputs/diffa-angry-remaining.log
bash run_diffa_angry_remaining.sh --mode summary
```

All results/vectors/configs are under
`outputs/diffa-angry-only-remaining/<dataset>`. Resume the same launch to skip saved
captures and answers. `--datasets multiarith ifeval` can prepare/run those two
independently while the SVAMP audio copy is pending; include all three after the
copy and audit. Their frozen plans remain unchanged when a later audit adds
SVAMP to the inventory.

```bash
.venv-diffa/bin/python pack_experiment_backup.py --only angry-remaining \
  --destination /workspace --samples 3 --sample-mb 100
```
