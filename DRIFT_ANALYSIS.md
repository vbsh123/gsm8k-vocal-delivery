# Drift from saved neutral/angry activation means

This analysis reads existing per-question activation caches. It never imports
the DIFFA loader, starts generation, installs hooks, or uses CUDA. PyTorch is
needed only to read tensors and calculate statistics on CPU.

On the original extraction/steering machine:

```bash
git pull --ff-only
nohup env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 \
  .venv-diffa/bin/python -u analyze_paired_drift.py \
  --collection outputs/jenny-svd-probe \
  --output outputs/jenny-drift-analysis \
  > outputs/jenny-drift-analysis.log 2>&1 < /dev/null &
tail -f outputs/jenny-drift-analysis.log
```

The input is `collection-config.json` plus `pairs/00000.pt`, etc., from the
matched neutral/angry collection used for SVD/probe fitting. Each pair file must
contain `neutral` and `angry` tensors of shape `[layers, groups, hidden_size]`
and the saved answer responses. The manifest embedded in the config supplies
reference answers; original WAV files are not needed. If the collection moved,
change `--collection`. An alternative all-forward collection can be analyzed
separately with another output folder; the script reports the averaging recipe.

A fitted `style-vectors.pt` alone is not enough: it contains pooled directions,
not the separate per-question neutral and angry means. Likewise, the temporal
collection under `jenny-large/temporal-vectors` stores differences for selected
targets and cannot reconstruct these neutral-to-angry norms and cosines. The
script rejects difference-only caches rather than fabricating missing values.

The outcome categories use the actual unsteered responses saved during collection:

- `recovery`: neutral wrong, angry correct.
- `regression`: neutral correct, angry wrong.
- `both_correct` and `both_wrong`.

These are natural-delivery outcomes, not steering outcomes. Parsing failures
count as incorrect under the existing scorer. This extraction collection was
filtered for emotion2vec agreement, so its own accuracy gap and outcome counts
need not equal the unfiltered benchmark's approximately six-point difference.

For each question, zero-based layer and group, the script measures:

- Cosine similarity between the saved neutral and angry means.
- Relative change: `100 * norm(angry - neutral) / norm(neutral)`.
- Absolute shift norm.

It then summarizes these within each outcome category. It also measures shift
direction coherence (the norm of the average unit shift), mean pairwise shift
cosine within each category, and the cosine between mean recovery and regression
shifts. Nonzero-shift sample counts are retained. Zero-norm vectors have missing
angles/relative changes rather than invented values.

Outputs in `outputs/jenny-drift-analysis`:

- `index.html`: self-contained layer tables for acoustic, semantic, combined
  (acoustic + semantic) and answer groups. Download the folder and open it.
- `pair-metrics.csv`: per-question/layer/group measurements and forward counts.
- `layer-summary.csv`: all group/outcome summary statistics.
- `outcome-comparison.csv`: recovery/regression contrasts and direction cosines.
- `summary.json`: counts, missing files, averaging, forward lengths and source hash.

Each pair contributes equally. Missing cache files are reported explicitly; a
partial collection is never described as complete. Terminal rankings show the
largest absolute recovery/regression drift contrasts, not established causal
sites. No p-values or claims of statistical significance are attached to this
exploratory layer ranking.

The cached vectors already average tokens and forwards. Consequently, this is
**drift of the pooled means**, not average token-level drift or the temporal
course of drift. Cancellation can hide differences. Matched forward counts do
not establish identical mask states, and neutral/angry generation may already
have diverged. Semantic token lengths can differ. These limitations are included
in the report: causal localization still requires a later intervention study.

No activation-patching runs are added by this script.

## Late-layer development sweep

`run_jenny_late_layers.py` tests the original winning checkpoint on both
evaluation cohorts: the 460 second-cohort neutrals used for the timing experiments
and the 258 eligible final-cohort neutrals. Extraction questions are excluded.
It uses each target
layer's own direction, with acoustic and answer groups separately, layers
29/30/31, strengths 0.5/1, and always/periodic2 schedules: 24 interventions.
Periodic2 acts on forwards 1, 3, 5, ... within each decoding block. Compatible
saved baselines are reused; every generation resets seed42. No extraction or
correctness filtering is added. This is a development sweep, not independent
confirmation. The default runs all 718 evaluation recordings; compatible saved
answers in each output folder are skipped.

```bash
nohup .venv-diffa/bin/python -u run_jenny_late_layers.py \
  > outputs/jenny-late-layers.log 2>&1 < /dev/null &
tail -f outputs/jenny-late-layers.log
.venv-diffa/bin/python run_jenny_late_layers.py --summary-only
```

Use `--prepare-only` for preparation without loading DIFFA, or `--vectors PATH`
if the original checkpoint moved. The launcher waits for the shared GPU lock
and active GPU work before running one evaluation process. Existing completed
answers in its own output folder are skipped on resume.

Drift on other cohorts requires their own saved neutral/angry per-question
activation means. Answer JSONL files and aggregate direction checkpoints cannot
replace those tensors. Run each collection into a separate analysis output
folder to preserve cohort comparisons; no new activation collection is launched
by this late-layer sweep.

The queue also appends L12 acoustic ×1 with the 20-forward-window checkpoint and
periodic2 steering, reported separately as `forward20`. Its checkpoint is read
from `outputs/jenny-large/temporal-evaluation/forward_windows/run_config.json`;
use `--forward20-vectors PATH` if it moved. Across both cohorts there are
17,950 intervention generations (718 × 25), plus any unavailable baselines.
Second-cohort results retain their original output paths. Final-cohort results
are stored under `third-cohort/`. `--summary-only` prints combined results first,
then the cohort breakdown. A completed run saves `combined-summary.json`.
If the old second-cohort launcher is already running, queue just the extension:

```bash
nohup .venv-diffa/bin/python -u run_jenny_late_layers.py --third-only \
  > outputs/jenny-late-layers-third.log 2>&1 < /dev/null &
tail -f outputs/jenny-late-layers-third.log
```

The extension waits for the shared GPU lock and runs only the final 258 questions;
it does not restart the second-cohort experiment. Both vector methods use the
same recordings and share compatible existing baselines.
The updated launcher first waits for `run_emotions_after_steering.sh`,
`run_jenny_third_accuracy.sh`, and their classification/accuracy workers to exit,
before acquiring the shared GPU lock. This lets that already-queued work take
priority without a lock-order deadlock. Log viewers are ignored.
