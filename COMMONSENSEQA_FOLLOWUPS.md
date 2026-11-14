# CommonsenseQA follow-ups

The same 500 generated questions, four Azure Jenny deliveries, and A–E task
prompt/scorer are used across DIFFA, Qwen and Kimi. Gold answers and transcripts
are never supplied during generation. Results are resumable and distinct from
the existing GSM8K files. Seed 42 resets before every generation.

## Evaluation machine: Qwen, Kimi, then DIFFA extraction/steering

```bash
git pull origin main
nohup .venv-diffa/bin/python -u run_commonsense_models.py \
  --release-diffa-cache --release-model-cache \
  > outputs/commonsenseqa-followups.log 2>&1 < /dev/null &
tail -f outputs/commonsenseqa-followups.log
```

The queue waits for existing pitch/speed and CommonsenseQA DIFFA work. It runs
Qwen, then Kimi, on all 2,000 clips. Afterward `steer_commonsense_mean.py` uses
DIFFA: 250 question IDs
are selected for extraction by a fixed seed42 shuffle, and the other 250 for
testing. Only angry/neutral pairs are used for extraction, with no classifier
or correctness filtering. L12 acoustic and L31 answer activations are averaged
within their token groups on each forward. Each question contributes the mean angry-minus-neutral
difference over its matched forwards; question means receive equal weight.
Each fitted mean is tested separately at strength 1 on held-out neutral recordings.
Compatible neutral benchmark answers are reused. There are at most 500
extraction generations and 500 new steered generations. No test question is
included in extraction. This tests a newly learned CommonsenseQA direction,
not transfer of the existing GSM8K vector.

Qwen and Kimi use the existing environments and model revisions from the GSM8K benchmark.
Each missing model's weights may need downloading again. At most one model is
loaded on the GPU. With the flags above, only downloadable LLaDA weights are
temporarily released before Qwen, and each Qwen/Kimi weight cache is removed after
all of that model's answers are saved. This avoids retaining three large models
on the 100 GB disk. Environments, audio, raw responses and learned vectors remain.
The original DIFFA component revision IDs and cached filenames are recorded
before release. The same snapshots are restored before steering, which then
runs offline to keep its cached-baseline model versions fixed.
Omit the release flags only if storage can accommodate the model caches.

Outputs:

- `outputs/commonsenseqa-angry-mean/`: frozen 250/250 split, extraction pair means
  and raw responses, `style-vectors.pt`, held-out baseline/steered answers,
  paired accuracy summary.
- `outputs/azure-commonsenseqa500/results-qwen-commonsenseqa.jsonl` and
  `results-kimi-commonsenseqa.jsonl`: model responses and scores.
- The same audio folder gets separate model settings and summaries with paired
  McNemar p-values and 95% bootstrap confidence intervals.
- `outputs/commonsenseqa-followups/status.json`: current stage.

Progress without GPU loading:

```bash
.venv-diffa/bin/python steer_commonsense_mean.py --summary-only
/venv/qwen-omni/bin/python evaluate_qwen_omni.py --task commonsenseqa --num-questions 0 --summary-only
/venv/kimi-audio/bin/python evaluate_kimi_audio.py --task commonsenseqa --num-questions 0 --summary-only
```

## Steering machine: narrow GSM8K strength sweep

```bash
git pull origin main
nohup .venv-diffa/bin/python -u run_L31_timing.py --strength-sweep \
  > outputs/jenny-L31-strengths.log 2>&1 < /dev/null &
tail -f outputs/jenny-L31-strengths.log
```

This keeps the original mean vector fixed and tries strengths 0.75, 1.25 and 1.5
on the existing 460+258 neutral recordings, always on. It reuses baselines and
shows saved strength1 and periodic2 results alongside the new conditions.
There are up to 2,154 new generations. Existing timing and +4.3 pp results are
unchanged. These cohorts are development data after the configuration search.

```bash
.venv-diffa/bin/python run_L31_timing.py --strength-sweep --summary-only
```
