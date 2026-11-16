# Rerun missing MultiArith and IFEval

`bash run_missing_experiments.sh` runs MultiArith first, then IFEval audio,
original four-delivery evaluation and half-collection/half-steering. It calls
the existing runners with their original defaults, without modifying their
prompts, seed policy, selection, generation, collection or scoring code.

- MultiArith: same pinned 600-row source, 580 unique questions, seed42 290/290
  split, neutral/angry collection, neutral/angry/steered-neutral test, L31
  answer x1, 512 tokens, 128 steps, 32-token blocks. Numeric accuracy primary.
- IFEval: same pinned 541 prompts and constraints, Azure Jenny intensity 1.5,
  neutral/angry/happy/sad recordings, all-four-at-most-30-seconds eligibility,
  original generic audio instruction, 1024 tokens and original decoder/seed42.
  Steering shuffles sorted eligible IDs with seed42, collects on the first half
  and evaluates the second half at L31 answer x1. Original official strict and
  loose checks. Last run yielded 508 eligible questions and a 254/254 split.

Regenerated Azure recordings may differ. Consequently IFEval's length-based
eligible subset and scores cannot be guaranteed identical without original
audio/eligibility. The original selection rule remains unchanged; no forced
508-question subset or replacement selection rule is introduced. MultiArith's
question split is fully determined by its pinned source and seed.

The queue waits for existing MOSS/Kimi and experiment queues on this machine.
Each original runner remains resumable. A failed stage stops subsequent stages.
The wrapper selects `.venv-diffa/bin/python` by default; override DIFFA_PYTHON
if necessary. Existing DIFFA weights and harness installation are required;
IFEval's original shell wrapper installs its checker dependencies. Export
AZURE_SPEECH_KEY and AZURE_SPEECH_REGION before starting.

From the repo root, after updating:

```bash
nohup bash run_missing_experiments.sh > missing-experiments.log 2>&1 < /dev/null &
```

The queue copies the five selected audio/result directories to `recovery-backups/`
outside outputs, every 30 seconds and after stages. Copies include WAV audio,
JSON/JSONL answers/settings, PT vectors/pair caches and logs within those folders.
Unchanged files are skipped; copies use atomic replacement. Removing a source
does not remove its backup. Backups never modify source files.
This protects against deleting outputs, but is still on the same disk: download
the backup to a separate machine/store. Additional disk use is dominated by
duplicate audio (previous MultiArith ~630 MB and IFEval ~1.6 GB).

Use `--task multiarith` or `--task ifeval` for only one experiment.
Use `--backup-root /path/outside/outputs` for another backup location.

```bash
bash run_missing_experiments.sh --mode summary
```
