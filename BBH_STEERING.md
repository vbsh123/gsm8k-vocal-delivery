BBH L31 steering: 125 extraction questions / 125 test questions
=============================================================

Run on the machine holding `outputs/azure-bbh-logical200` and the completed
`outputs/bbh-logical200-diffa2-harness-chunked` benchmark.

```bash
nohup .venv-diffa/bin/python -u steer_bbh_mean.py \
  > outputs/bbh-steering125.log 2>&1 < /dev/null &
tail -f outputs/bbh-steering125.log
```

Azure credentials (`AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`) must be exported
in that shell. Existing audio is linked into the new source folder; only
neutral and angry recordings for the 50 unused questions are generated (100
clips). The original 200-question source and benchmark are unchanged.

The existing seed42 shuffled order of all 250 official three-object questions
defines the split: first 125 extraction, remaining 125 evaluation. No outcome
or emotion-classifier filtering. There is no question overlap between halves.

Collection uses the unchanged BBH prompt and full-audio chunked Whisper method
from the completed benchmark. On each L31 forward, average the last 512 answer
token positions. For each question, keep the common neutral/angry forward
count and average angry-minus-neutral differences. Average these 125 question
vectors equally. Add that fixed mean to L31 answer tokens at strength 1 on
every forward of held-out neutral generation. This uses the shared steering
hook; one model is loaded for collection and evaluation.

All generations reset seed42 and use 512 answer tokens, 128 steps, 32-token
blocks, temperature 0 and fast_dllm, with 30-second Whisper chunks followed
by the same global adapters as before. The requested final line remains
`The answer is (<letter>).`

Evaluation has three conditions on the test half: neutral, actual angry
delivery, and steered neutral. Matching saved benchmark answers are reused;
missing references are generated with the same recipe. The pinned upstream
harness scorer cache is reused when available. Both strict-match and
flexible-extract scores, paired McNemar p-values and 95% paired bootstrap
intervals are reported. This is our spoken adaptation, not a full official
text harness inference run. The split is exploratory because aggregate
delivery results on the earlier 200 questions were already examined.

The script waits for active SVAMP/BBH jobs, then the shared GPU lock and GPU
availability. Extraction pairs and evaluation answers resume individually.
It normally needs 250 extraction generations, 100 new neutral/angry reference
answers, and 125 steered answers: 475 model generations if the existing 200
answers are complete. Existing audio links require retaining the original folder.

Check all completed evaluation results while it runs:

```bash
.venv-diffa/bin/python steer_bbh_mean.py --mode summary
```

Files:

- `outputs/azure-bbh-logical250-steering/`: all 250 questions, linked old audio,
  and 100 new clips.
- `outputs/bbh-logical250-angry-mean/split.json`: fixed split and recipe.
- `outputs/bbh-logical250-angry-mean/pairs/`: resumable per-question collection.
- `outputs/bbh-logical250-angry-mean/style-vectors.pt`: learned vector.
- `outputs/bbh-logical250-angry-mean/results.jsonl`: held-out raw responses.
- `outputs/bbh-logical250-angry-mean/summary.json`: harness scores and comparisons.
