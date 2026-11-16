# Kimi GSM8K nightly run

Uses all **1,269** questions from the three main Azure cohorts (GSM8K test indices
50–1318). The old 718 DIFFA evaluation questions were a subset, separate from its
collection cohort. This experiment makes a fresh seed42 split: **634 collection /
635 held-out test**. It retains all recordings, including ones over 30 seconds,
through Kimi's native encoders. No outcome or classifier filtering.

The checked-in question bank contains only original wording, numeric gold and IDs
from the archived Azure plans; no model responses or correctness-based selection.
Its hash and the complete split are frozen before TTS/inference. L17 answer x1 is
fixed from BBH, with no layer search and no reuse of DIFFA vectors.

```bash
git pull --ff-only
nohup bash run_kimi_gsm8k.sh > outputs/kimi-gsm8k-L17.log 2>&1 < /dev/null &
tail -f outputs/kimi-gsm8k-L17.log
```

Export `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` before launching if audio is
missing. Compatible Jenny intensity1.5 clips in the four original Azure folders
are linked into a separate audio folder; originals remain intact. Missing clips
are generated with resume. This can require up to 2,538 clips. Links depend on
their original audio files, so preserve those folders when backing up.

Audio preparation runs on CPU. Model work waits for MultiArith, prior BBH queues,
and the shared GPU lock. One cached Kimi-Instruct model performs 1,268 collection
generations and 1,905 test generations (neutral, angry, and neutral steered with
its own angry-minus-neutral mean). Collection and test use the same math `####`
prompt, seed42 and native greedy 512-token decoder. Mean vector extraction and
steering use the same shared implementation as Kimi MultiArith. Small L17 audio
differences are saved for reference; only the answer vector is applied.

```bash
.venv-diffa/bin/python run_kimi_gsm8k.py --mode summary
```

Outputs: `outputs/kimi-gsm8k634-635-L17/` (selection, config, pair caches, vector,
raw responses, scorer provenance, rescored responses and summary). Audio and
listening page: `outputs/azure-kimi-gsm8k1269/`.

Primary comparison: numeric correctness, steered neutral versus neutral on all
635 test questions. Scoring uses installed upstream GSM8K strict/flexible filters
and exact match, plus additional numeric equivalence and format diagnostics.
Reports include matched counts, exact McNemar p-values and paired bootstrap 95%
intervals. Secondary comparisons are unadjusted. This is audio-adapted inference,
not the full official text-harness prompting protocol. No base checkpoint or
additional model is downloaded. Resume retains completed pairs and test answers.
