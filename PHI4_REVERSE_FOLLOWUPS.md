# Phi reverse steering: L30, then chosen-layer continuation

First collect L30 on the same400 questions and evaluate on the same400 tests:

```bash
git pull --ff-only
nohup bash run_phi4_gsm_reverse400.sh --layer 30 \
  > outputs/phi4-gsm-reverse400-400-L30.log 2>&1 < /dev/null &
tail -f outputs/phi4-gsm-reverse400-400-L30.log
```

Same original Phi reasoning prompt, seed42 and strengths -0.5/-1. Fresh L30
native generation-forward means, equal weight per collection question.
Compatible400 neutral baseline answers are reused from L17, after checking
prompt/model, generation settings, question/voice, reference and audio hash.
At most800 collection and800 steered test generations. Completed answers resume.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u steer_phi4_gsm_reverse.py --layer 30 --mode summary
```

## Later continuation: choose layers

Choose ONE command. Both strengths are always retained:

```bash
# Both layers
nohup bash run_phi4_reverse_followups.sh 17 30 > outputs/phi4-reverse-followups.log 2>&1 < /dev/null &
# Or L30 only
nohup bash run_phi4_reverse_followups.sh 30 > outputs/phi4-reverse-followups.log 2>&1 < /dev/null &
# Or L17 only
nohup bash run_phi4_reverse_followups.sh 17 > outputs/phi4-reverse-followups.log 2>&1 < /dev/null &
```

```bash
tail -f outputs/phi4-reverse-followups.log
```

The queue waits for current400/400 jobs (including L30), model and instruction
queues. Each inference stage additionally acquires the shared GPU lock and
waits for active GPU processes. No jobs are stopped. A layer's process exits
before the next process waits for GPU, avoiding waits on its own CUDA context.

All chosen GSM continuations run FIRST, then MultiArith. Choose after reviewing
L30 if desired; choices remain fixed for that queue.

### GSM

1269 questions =400 collection +400 original test +469 remaining. Keep each
chosen layer's frozen vector and all original1200 answers; add only missing
answers on469 neutral recordings. Combined reports cover869 non-collection
questions,2607 answers per layer. Original data are never edited. Imported
answers preserve source config/provenance, and source hashes are frozen.
Between layers, matching neutral baselines are also reused. Original400/400
runs must be complete before their continuation. No vector recollection.

Outputs: `outputs/phi4-gsm-reverse400-869-L17` and/or `...-L30`.

### MultiArith

Pinned600 rows deduplicate to580 unique questions. Existing seed42 split is
290 collection /290 test, no outcome/classifier/duration filtering. Full audio
is reused from `outputs/azure-multiarith580`; no TTS is launched. Own-task
angry-minus-neutral vectors are collected using the same original Phi math
prompt, native greedy decoder, seed42 and generation-forward averaging.

Both L17/L30 means are observed in the SAME580 neutral/angry collection
generations, cached once in `outputs/phi4-multiarith-reverse290-shared`.
The two layers see identical native histories; emotions use their own histories.
Only selected layers are evaluated, each on290 neutral questions at baseline,
-0.5 and -1. Neutral baselines are reused across selected layers after checks.

Outputs: `outputs/phi4-multiarith-reverse290-290-L17` and/or `...-L30`.

```bash
# Change --layer to17 or30 as desired; CPU reports only.
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u phi4_reverse_followups.py --stage summary-gsm --layer 30
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u phi4_reverse_followups.py --stage summary-multiarith --layer 30
```

Saved: configs, native collection responses/means, vectors, raw test answers,
runtime, official filter provenance and reports. Partial work resumes.
Collection never uses test outcomes. These follow-ups and layer choices were
informed by earlier outcomes/geometry and remain exploratory. Scoring uses
installed upstream GSM8K filters/metric, extra numeric equivalence and separate
format extractability. On MultiArith this is an audio adaptation, not an
official MultiArith harness task. P-values and paired95% intervals unadjusted.
