# Strength and vector-transfer sweeps on existing DIFFA winners

Only these six task settings are included: **GSM8K, SVAMP, MultiArith, IFEval,
BBH logical deduction three objects, and BBH logical deduction five objects**.
BBH variants are separate tasks within the same suite. Kimi/MOSS vectors are not
compatible with these DIFFA checkpoints and are not included.

Both queues use the already successful **L31, answer positions, always schedule,
fixed angry-minus-neutral mean**. They reuse collected vectors exactly, without
normalization, recollection, new TTS, outcome filtering or prompt changes. Vector
magnitudes can differ across datasets; x1 means one original raw vector unit.

## 1. Own-vector strengths

```bash
nohup bash run_winning_strengths.sh \
  > outputs/winning-strengths.log 2>&1 < /dev/null &
tail -f outputs/winning-strengths.log
```

Each dataset uses its own vector and **its existing evaluation cohort** at:

```
-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
```

Compatible original baselines and x1 answers are reused. The existing aligned
GSM8K strength sweep is also checked for compatible completed rates. There are
six dataset jobs, nine rates each; with complete original runs, only seven new
rates per question remain. This is a substantial queue, not a small smoke test.

## 2. Every vector on every OTHER winning task

```bash
nohup bash run_winning_transfer.sh \
  > outputs/winning-transfer.log 2>&1 < /dev/null &
tail -f outputs/winning-transfer.log
```

There are **6 × 5 = 30 off-diagonal source-vector/target-task jobs**, at **x1**.
Each target keeps its own prompt, answer budget, recorded voice/audio, evaluation
questions, gold references, and evaluator. No own-vector diagonal is repeated.
For example, SVAMP-vector -> IFEval keeps the IFEval prompt and 1,024-token budget;
IFEval-vector -> GSM8K keeps the aligned GSM8K prompt and 512-token budget.

Each job reuses the target's original neutral baseline. Existing BBH3-vector ->
BBH5 transfer answers are also reused when their vector hash, question/voice,
audio, prompt and generation recipe match the target's fixed test cohort.

## Queueing, resume, and preflight

The shell wrappers wait for original winner input-generating/evaluation queues.
Both new queues then use the shared `outputs/jenny-task-vectors.lock`, so they can
be launched together and will run serially. Each holds that lock for its whole
queue and loads **one DIFFA model** for all remaining jobs. They never remove
model caches, audio, results, environments, or source vectors.

Before loading a GPU model, preflight validates **all six** original cohorts,
source vectors, prompts, output budgets, collection IDs, audio hashes and files.
Missing inputs are reported together; jobs are not silently dropped. The exact
planned and pending generation counts are printed. Individual answers are saved
and flushed immediately and resume independently; compatible completed answers
are not regenerated. Partial trailing lines are repaired only in new queue output
files, never in original results. Different frozen settings require a new output.

Default input locations are in `winning_diffa_catalog.json`. Edit a copy and pass
`--catalog /path/to/catalog.json` if the recovered files have different paths.
An optional `vector` entry overrides a relocated vector path; optional `reuse_runs`
lists additional historical answer directories. Changed historical prompts or
vector hashes cannot be relabeled as compatible results.

CPU preparation without waiting for the current GPU queue:

```bash
.venv-diffa/bin/python run_winning_strengths.py --mode prepare
.venv-diffa/bin/python run_winning_transfer.py --mode prepare
```

These write separate frozen manifests and copy compatible answers into the new
output folders. They do not load a model or run generations. If source IFEval or
another experiment is still collecting, use the shell wrappers and let them wait.

## Evaluation and progress

```bash
.venv-diffa/bin/python run_winning_strengths.py --mode summary
.venv-diffa/bin/python run_winning_transfer.py --mode summary
```

Results live in:

- `outputs/winning-strengths/<dataset>/`
- `outputs/winning-transfer/<target>__from__<source>/`

Each directory contains frozen settings, full raw responses, freshly rescored
answers, scorer provenance, and per-condition summaries. Root files include
`queue_manifest.json`, `summary.json`, and `strength-summary.csv` or
`transfer-matrix.csv`. The transfer CSV contains only x1 off-diagonal comparisons;
per-job summaries retain the baseline checks too. The aggregate CSVs contain the
target's primary metric and do not pool incompatible metrics across tasks.

Scoring is unchanged in substance:

- Math: installed upstream GSM8K strict/flexible filters and exact-match metric;
  additional numeric equivalence and format-compliance diagnostics. SVAMP also
  retains the authors' numeric tolerance diagnostic. These are spoken adaptations,
  not full official text-harness inference or dedicated MultiArith/SVAMP tasks.
- BBH3/5: their matching installed upstream CoT zero-shot filters and metrics.
- IFEval: pinned official strict/loose prompt and instruction checkers, preserving
  the original constraints. No semantic/factual accuracy metric is invented.

Every answer comparison uses the same target question and voice against its
neutral baseline. Reports include wins/losses, exact McNemar p-values and 95%
question-paired bootstrap intervals. IFEval pooled instruction rates are
descriptive; inferential tests use prompt-level outcomes. All sweep comparisons
are **exploratory and unadjusted**, not independent confirmatory tests. There is
no automatic best-strength selection or best-strength test-set performance claim.
