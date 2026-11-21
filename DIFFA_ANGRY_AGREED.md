# Emotion2vec audit and filtered angry vectors

This is a separate DIFFA follow-up. Original vectors, evaluations, prompts,
splits and results remain saved. The default steering datasets are SVAMP,
MultiArith, BBH3, BBH5 and IFEval; GSM is optional because its original vector
already used emotion agreement. Happy audio is audited but no happy vector is
collected by this runner.

## 1. Audit existing recordings first

```bash
git pull --ff-only
nohup bash run_diffa_angry_agreed.sh --mode audit \
  --datasets gsm8k svamp multiarith bbh3 bbh5 ifeval \
  > outputs/diffa-angry-agreed-audit.log 2>&1 </dev/null &
```

This inventories available Jenny neutral/angry/happy audio for the existing
collection and test IDs, plus other surviving question IDs with the same dataset
ID prefix. Other IDs are audit-only and never enter vector fitting or testing.
It does not generate TTS or answers. Missing happy or
angry recordings are reported; none are fabricated. Original neutral and angry
clips are pinned by saved hashes where available. All surviving variants are
included in classifier diagnostics, deduplicated by audio hash.

Whole-recording emotion2vec+ Large predictions are reused only with exact audio
SHA256, matching model name, successful status and whole-recording processing.
Legacy labels without these receipts are reclassified. New classification uses
`.venv-emotion/bin/python` (override with `EMOTION_PYTHON`) and waits for the
shared DIFFA GPU queue. `--device cpu` avoids GPU waiting. Only classifier work
runs at this stage; no vector collection or steering starts automatically.

```bash
bash run_diffa_angry_agreed.sh --mode audit-summary
```

The compact table shows dataset, split (`all`, `collection`, `test`, and any
audit-only `other` questions), requested style, expected questions,
available unique audio, classified clips, agreed clips, match rate, errors and
pending clips. It also reports eligible neutral/angry *pairs* within each
collection half. Style-level percentages alone do not give the paired count.
The final dataset line reports neutral/angry and neutral/happy paired counts and
percentages within that original collection half, plus the intersection where
all three labels agree. Pending/error predictions cannot count as agreement;
inspect classifier coverage before interpreting these paired percentages.
Multiple surviving variants can make unique available audio exceed question
count; selection uses the original pinned neutral/angry recordings only.

Files are `outputs/diffa-angry-agreed/audit/cohorts.json`,
`audio_inventory.jsonl`, `results-emotion2vec.jsonl`, `dataset-summary.json`
and `dataset-summary.csv`. Classifier failures remain explicit; collection
preflight requires successful labels for every available collection pair,
so pending/error labels cannot silently become exclusions. Missing original
collection audio must be restored before collection.

Inspect these percentages and pair counts before launching the next stage.
There is no automatic definition of a "good enough" agreement percentage.

For the follow-up requiring more than50% paired happy agreement on every dataset,
include GSM explicitly and request the gate:

```bash
bash run_diffa_angry_agreed.sh --mode prepare \
  --datasets gsm8k svamp multiarith bbh3 bbh5 ifeval --require-happy-pairs-above 50
nohup bash run_diffa_angry_agreed.sh --mode run --stage primary \
  --datasets gsm8k svamp multiarith bbh3 bbh5 ifeval --require-happy-pairs-above 50 \
  > outputs/diffa-angry-agreed-all-primary.log 2>&1 </dev/null &
```

The gate uses the original collection half, requires both neutral and happy to
agree, and rejects pending/missing/error labels before collecting any dataset.
It does not silently skip failing datasets. Exactly50% does not pass. This gate
is independent of angry-pair selection: angry vectors still require neutral/angry
agreement and test cohorts remain unchanged. Including GSM recollects its already
agreed direction using the same follow-up runner, rather than overwriting or
reinterpreting the original saved vector. Strengths and transfer are still separate
explicit stages after inspecting these primary results.

## 2. Collect and test angry at +1

```bash
# CPU-only selection and baseline/evaluator preflight.
bash run_diffa_angry_agreed.sh --mode prepare

# Launch only after inspecting the audit.
nohup bash run_diffa_angry_agreed.sh --mode run --stage primary \
  > outputs/diffa-angry-agreed-primary.log 2>&1 </dev/null &
```

Defaults: SVAMP, MultiArith, BBH3, BBH5, IFEval. Pass `--datasets` to restrict
them. The selection rule requires neutral->neutral and angry->angry predictions
inside the original collection half. It never takes replacement questions from
the held-out half, so the final collection can contain fewer than half of all
questions. Test questions are never classifier-filtered. No outcome-based
selection is used. With no agreed pairs, pending labels, missing original audio
or missing compatible neutral baseline answers, preflight stops before loading
DIFFA. Inventory is frozen for selection; changing it later requires a separate
output root rather than modifying an ongoing experiment.

Collection uses the saved collection prompt, budget, seed42, fast-dllm128 steps,
block32, matched-forward token means and equal per-question weighting. New
streaming aggregates retain raw collection responses and all32 layers/four token
groups without storing large per-pair activation caches. Vectors are raw
angry-minus-neutral means. The test adds the L31 answer vector at +1 to neutral
input using the original target prompt and budget. Original neutral test answers
are reused; no neutral test inference or unsteered angry/happy delivery inference
is scheduled. Passive neutral *collection* forwards are necessary.

One persistent DIFFA model serves the datasets sequentially after waiting for the
shared GPU queue. The old happy pipeline is not stopped automatically. Checkpoints
and answers are resumable, and different experiments use separate output folders.

```bash
bash run_diffa_angry_agreed.sh --mode summary --stage primary
```

Outputs per dataset: `run_config.json` with exact selected/excluded-pool IDs and
classifier receipts, `collection-stream.pt`, `collection-results.jsonl`,
`style-vectors.pt`, `layer-drift.csv`, and `primary/` with raw answers, scoring
provenance, paired metrics and matched mean/median response tokens.

## 3. Optional later stages, not part of the default run

Only after inspecting the primary results:

```bash
nohup bash run_diffa_angry_agreed.sh --mode run --stage strengths \
  > outputs/diffa-angry-agreed-strengths.log 2>&1 </dev/null &

# Separate explicit launch; waits for the same shared GPU queue.
nohup bash run_diffa_angry_agreed.sh --mode run --stage transfer \
  > outputs/diffa-angry-agreed-transfer.log 2>&1 </dev/null &
```

Strengths default to -2,-1.5,-1,-0.5,0,0.5,1,1.5,2; override with `--strengths`.
The primary +1 answers are reused only when the exact filtered vector matches.
Transfer runs every off-diagonal selected dataset pair at +1 with target prompts,
budgets, scoring and neutral baselines preserved. All selected datasets must have
completed filtered-vector collections before these stages. Defaults operate on
the five new datasets; include GSM explicitly in audit/primary first if a new,
identically audited GSM direction is desired in that transfer matrix.

```bash
bash run_diffa_angry_agreed.sh --mode summary --stage strengths
bash run_diffa_angry_agreed.sh --mode summary --stage transfer
.venv-diffa/bin/python pack_experiment_backup.py --only angry-agreed \
  --destination /workspace --samples 3 --sample-mb 100
```

This is an exploratory classifier-selected experiment. Emotion2vec labels do
not establish human perceptual ground truth, and filtering need not improve
accuracy, transfer or emotion specificity. Compare filtered and original vectors
on the unchanged test cohorts and report the number of contributing pairs;
different sample counts remain a possible explanation for any difference.
