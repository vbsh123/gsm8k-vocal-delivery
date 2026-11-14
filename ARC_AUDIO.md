ARC-Challenge provides grade-school science multiple-choice questions. This experiment uses 500 questions sampled with seed 42 from the official 1,172-question test split. Selection requires four choices and excludes explicit visual references/math markup before sampling. The selection is independent of all model outputs. Original option order is retained; numeric labels are remapped to A–D, including the gold label. The complete source parquet is 204 kB, checksum-verified and retained with source IDs and original labels. No ARC data is used to fit a vector in this run.

On the evaluation machine, leave Azure credentials set and run:

```bash
git pull --ff-only
nohup bash run_arc_night.sh > outputs/arc-night.log 2>&1 < /dev/null &
tail -f outputs/arc-night.log
```

The queue generates neutral/angry/happy/sad Azure Jenny audio at the existing intensity 1.5 (happy uses cheerful), then waits for existing GPU jobs and the shared GPU lock. It evaluates DIFFA on every eligible recording plus each neutral recording with the existing CommonsenseQA L31 answer mean at strength 1. This is frozen-vector transfer from CommonsenseQA to ARC, not a newly fitted ARC vector. With all recordings under 30 seconds, this is 2,000 unsteered + 500 steered generations. Longer clips are listed in run_config.json and excluded without cropping; each comparison uses completed matched question/voice pairs.

The original CommonsenseQA vector must be at `outputs/commonsenseqa-angry-mean/style-vectors.pt`. To use the GSM8K original mean instead, copy that existing vector to this machine and pass its path explicitly to `evaluate_arc_audio.py --vectors PATH --output outputs/arc-GSM-vector`. Keep separate result folders when changing vectors. To evaluate delivery only, omit --vectors. To generate only neutral/angry, pass `--styles neutral angry` to the generator in a separate source folder.

All inference uses the shared DIFFA loader/generator: seed 42 reset before every generation, fast_dllm, 128 steps, 32-token blocks, 512 output tokens, temperature 0. The ARC prompt correctly describes four choices. Both baseline and steering use this identical prompt. One model is loaded; hooks add the fixed L31 answer vector only to answer positions.

The report separates:

- Answer correctness, using the corrected custom relaxed choice parser v2, including bare letters and exact option text.
- Format compliance with the requested `Final answer: A`–D syntax.
- Correct answers that also comply with that format (the old strict metric).

The parser is our documented scorer, not an official ARC likelihood harness. Full responses are retained for independent/manual inspection. Paired gains, wins/losses, exact McNemar p-values and 200,000-resample paired bootstrap confidence intervals are reported for answer correctness and strict accuracy. P-values/intervals are unadjusted. Successful audio downloads and saved generations resume; no weights or audio are deleted.

Mid-run summaries (CPU only):

```bash
.venv-diffa/bin/python evaluate_arc_audio.py --summary-only
```

The existing DIFFA environment pins compatible pyarrow 19.0.1. If it is missing, install that package alone into .venv-diffa; no Torch/model changes are required.

On the steering machine, inspect the existing adaptive nightly run before adding another search:

```bash
.venv-diffa/bin/python run_L31_adaptive.py --summary-only
cat outputs/jenny-L31-adaptive/status.json
```

Once its evaluation files exist, independently rescore its saved answers without rerunning DIFFA:

```bash
.venv-diffa/bin/python rescore_audio_steering.py --task gsm8k \
  --results outputs/jenny-L31-adaptive/evaluation/second/results.jsonl \
            outputs/jenny-L31-adaptive/evaluation/third/results.jsonl \
  --conditions same-state-mean-L31-a1 adaptive-L31-a0.5 adaptive-L31-a1 \
  --output outputs/jenny-L31-adaptive/scoring-audit
```

The latter command compares current strict scores with upstream GSM8K flexible extraction, both string exact match and numeric equivalence. Use completed matched pairs only; if one cohort has not started yet, pass only the file that exists. The 718 questions have already informed method selection, so additional comparisons on them are development experiments.

The adaptive experiment has now completed and did not beat the original mean under strict scoring. A useful new steering-machine nightly run is to rescore those existing answers, then compare the original direction against its reverse and three random directions with the exact same norm:

```bash
git pull --ff-only
nohup bash run_steering_controls_night.sh > outputs/steering-controls-night.log 2>&1 < /dev/null &
tail -f outputs/steering-controls-night.log
```

This does not repeat adaptive extraction or adaptive inference. It creates four small fixed vector checkpoints on CPU, then evaluates each control at L31 answer strength 1 on the existing 460+258 questions, reusing the saved baseline. At most 2,872 new generations. Random construction uses an independent CPU generator seeded 42; inference is reset to seed 42 for every generation. Exactly the same prompt, token positions, vector norm and decoding settings are used. Results test direction specificity, not a new improvement method. The final report compares original and control directions under all three GSM8K scorers, using actual upstream extraction code. It also reports direct paired comparisons of each control against the original learned direction, rather than inferring a difference from separate p-values against the unsteered baseline. Completed control answers resume.

```bash
.venv-diffa/bin/python run_L31_direction_controls.py --summary-only
```
