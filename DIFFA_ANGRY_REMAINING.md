# Remaining angry-only agreed collections

This runs SVAMP, MultiArith and IFEval sequentially with one DIFFA model, after
waiting for the shared GPU lock and idle GPU. It is separate from the BBH3/BBH5
125-pair experiment. Only the angry clip must be classified as angry by
emotion2vec; neutral recognition is not a selection requirement.

| Dataset | New collection | Remaining test | Original eligible collection clips |
|---|---|---|---|
| SVAMP | 493 | original 500 | 493/500 |
| MultiArith | 290 | 290 | 272/290 |
| IFEval | 254 | 254 | 248/254 |

For SVAMP, use all493 angry-recognized original collection pairs, leave the seven
rejected collection questions unused, and keep all500 original test questions.
No SVAMP angry test audio is required. For MultiArith/IFEval, keep eligible original
collection questions first, then fill the half from eligible original test
questions using seed42. Evaluate every remaining question
without classifier filtering. The new split is exploratory and fixed before
generation. There is no selection by response correctness. Original experiments
remain saved. Original token budgets, prompts, matched-forward raw mean extraction,
seed42,128 steps/block32 and L31 answer+1 are preserved; IFEval uses1024 tokens,
math uses512. Compatible neutral baselines are reused; missing new-test baselines
are generated. Reports include a direct original/new comparison on questions
held out from both vectors, plus primary results on the whole new test half.

## Prepare from existing labels and launch

```bash
git pull --ff-only
bash run_diffa_angry_remaining.sh --mode prepare
nohup bash run_diffa_angry_remaining.sh \
  > outputs/diffa-angry-remaining.log 2>&1 </dev/null &
```

Preparation reads the existing completed audit at `outputs/diffa-angry-agreed/audit`
when no dedicated audit exists. There is no required audio copy, TTS or new
classification. Optional `--mode audit` still refreshes an isolated
`outputs/diffa-angry-only-remaining/audit` if needed. Preparation validates the
required audio/angry-label coverage before model startup. SVAMP explicitly requires
the493 recognized original pairs and rejects an old resplit configuration in the
same output directory. No happy vectors, strengths or transfer run automatically.

```bash
tail -F outputs/diffa-angry-remaining.log
bash run_diffa_angry_remaining.sh --mode summary
```

All results/vectors/configs are under
`outputs/diffa-angry-only-remaining/<dataset>`. Resume the same launch to skip saved
captures and answers. `--datasets` can restrict the sequence to selected tasks.

```bash
.venv-diffa/bin/python pack_experiment_backup.py --only angry-remaining \
  --destination /workspace --samples 3 --sample-mb 100
```
