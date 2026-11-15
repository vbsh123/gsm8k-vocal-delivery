# Tracking Shuffled Objects follow-up

Complete the same seeded three-object tracking cohort from 200 to all 250
questions in neutral, angry and happy. Existing audio and compatible saved
answers are reused; the first stage needs at most 150 new clips and answers.
The original 200-question results remain in their existing folder.

Next, use the first 125 questions in the fixed seed-42 shuffled order for
angry-minus-neutral mean collection, and the other 125 for evaluation.
Collection averages matched forwards within each question, then gives each
question equal weight. Steering uses L31 answer tokens at strength 1, with the
same tracking prompt, seed 42, fast_dllm decoder and installed upstream tracking
task scorer. Neutral and angry test answers are reused; only the 125 steered
test answers require new evaluation generations. No outcome or classifier filter.

This is exploratory: delivery results on 200 questions were already inspected.
The 250-question delivery table and 125-question steering table overlap and
must not be counted as independent cohorts. Long audio uses the existing
chunked Whisper/global-adapter path for every condition.

On the machine holding `outputs/azure-bbh-reasoning200` and
`outputs/bbh-reasoning200-diffa2`, with Azure credentials exported:

```bash
git pull --ff-only
nohup .venv-diffa/bin/python -u run_bbh_tracking250.py \
  > outputs/bbh-tracking250.log 2>&1 < /dev/null &
tail -f outputs/bbh-tracking250.log
```

The shared GPU lock serializes inference with other experiments. Separate
stage processes release the delivery model before collection starts. Rerunning
the command resumes missing clips, collection pairs and evaluation answers.

```bash
.venv-diffa/bin/python run_bbh_tracking250.py --stage summary
```

Audio: `outputs/azure-bbh-reasoning250/tracking_shuffled_objects_three_objects`.
Delivery results: `outputs/bbh-tracking250/deliveries`.
Split, pair tensors, `style-vectors.pt` and steering results:
`outputs/bbh-tracking250/steering`.
