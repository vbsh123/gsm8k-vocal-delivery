# Queued Phi BBH combinations and IFEval all-layer sweep

Run on the Phi machine (`C.59733963`). The CPU launcher uses `.venv-diffa`; native workers use `.venv-bbh-phi4`. Existing pinned Phi checkpoint, processor, prompts, seed42, greedy decoding and audio are retained. No new model/environment setup is required.

## One night launcher

```bash
git pull --ff-only
nohup bash run_phi4_bbh_then_ifeval.sh > outputs/phi4-bbh-ifeval-nightly.log 2>&1 </dev/null &
tail -F outputs/phi4-bbh-ifeval-nightly.log outputs/phi4-bbh-combo.log outputs/phi4-ifeval-all-layers.log
```

Waits for the shared GPU lock and all current GPU processes to finish. Then runs:

1. BBH3 and BBH5 combined steering, one persistent model worker per task in parallel.
2. IFEval all32 layers at−1, audio and answer positions separately, **three persistent model workers in parallel**.

The two groups run sequentially, so they do not load five Phi models together. Each group owns the shared GPU lock for its full lifetime. Current SVAMP/IFEval runs are not interrupted. Worker logs appear only after GPU admission. Launch the night command once, not together with the individual launchers below.

## BBH combined only

Each BBH uses its own original125 collection/125 test split and own raw angry-minus-neutral vectors. Apply **L0 audio−1 during prefill + L30 answer−2 at each last-prediction forward**, within one native generation. Reuse compatible collected vectors and existing neutral test responses. Missing vector captures are completed from the collection half; no standalone L0-only or L30-only test evaluations are launched. Each task generates only125 combined test answers, with512-token budget. Parent preflight fails rather than regenerating a missing neutral test baseline.

BBH workers reuse the standard per-layer/group collection cache in `outputs/phi4-sweeps/collection-cache`. Separate source vector receipts and combined results are under `outputs/phi4-bbh-combined/<bbh3|bbh5>/`. One Phi model per worker is reused for collection and generation; no second model is loaded for its combined stage.

```bash
# Standalone alternative; already included in the night command.
nohup bash run_phi4_bbh_combo.sh > outputs/phi4-bbh-combo.log 2>&1 </dev/null &

# CPU summary of both BBHs, including partial matched results.
bash run_phi4_bbh_combo.sh --mode summary

tail -F outputs/phi4-bbh-combined/worker{0,1}.log
```

Scoring uses the installed upstream three/five-object BBH tasks and their original saved-answer filters. Reports flexible/strict choice accuracy, strict-format extractability, matched response-token mean/median, exact paired McNemar p-values and95% paired bootstrap intervals. Numeric math scoring is not applied to BBH choices.

## IFEval all layers, both token groups

Keeps the original254 collection/254 test split and1024-token native Phi budget. All32 decoder layers0–31 are evaluated at raw strength−1 for each group (`audio`, `answer`), **64 distinct settings**, up to16,256 new steered test responses. Compatible existing neutral and angry test answers are reused; missing baselines cause an error before GPU admission rather than new baseline inference. There is no outcome-based selection.

Collection is split85/85/84 questions across the three workers. Each missing neutral/angry pair is generated once with passive capture of **all32 layers and both groups together**. Answer means average every native last-prediction forward including prefill; audio means average audio positions on prefill only. Pair differences are equally weighted across questions. Parent merges the shared all-layer vectors after all three workers reach the barrier. Workers stay loaded, then evaluate disjoint assignments of22/21/21 settings; they do not repeat the same layers/jobs or reload Phi per setting. Torch allocation cap is15GiB per worker to accommodate three A40 workers.

Raw per-question traces/means and merged directions are under `outputs/phi4-ifeval-all-layers-minus1/collection-cache/all-layers/ifeval`. Setting configs/results/vectors are under `.../ifeval/L<layer>-<group>-a-1`. Runtime collection receipts and worker assignments retain question IDs, audio hashes, model recipe and source provenance. Collection can resume after interruption; a new launch token prevents stale barrier files from releasing workers early. Parent detects failed workers and terminates remaining children; SIGTERM during the worker phase also cleans them up.

```bash
# Standalone alternative; already included in the night command.
nohup bash run_phi4_ifeval_all_layers.sh > outputs/phi4-ifeval-all-layers.log 2>&1 </dev/null &

# Full CPU summary: official IFEval prompt/instruction checks, lengths and paired tests.
bash run_phi4_ifeval_all_layers.sh --mode summary

# Compact strict-prompt results and per-setting progress.
bash run_phi4_ifeval_all_layers.sh --mode summary 2>/dev/null | \
  awk '/^ifeval:/{print} /neutral-ifeval-.* \/strict_prompt:/{print}'

tail -F outputs/phi4-ifeval-all-layers-minus1/worker{0,1,2}.log
```

Older single-layer IFEval and SVAMP outputs remain separate. Both new result roots are included in the existing Phi evidence backup group. Sweeps and comparisons are exploratory/unadjusted; report all settings rather than presenting selected-layer p-values as confirmatory. CPU tests verify disjoint assignments, actual parallel child-process barriers, vector/group slicing, generation-only BBH behavior and correct upstream scorers. Real Phi GPU inference is performed on the remote machine.
