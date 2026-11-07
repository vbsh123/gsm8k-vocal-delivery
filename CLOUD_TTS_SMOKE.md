# Cloud TTS smoke

`cloud_tts_smoke.py` generates one unchanged math question in neutral, angry,
happy, and sad deliveries. It uses a fixed voice within each provider. No GPU or
extra Python packages are needed. API calls use your provider account credits.

Set the credentials in your terminal (do not commit keys):

```bash
export AZURE_SPEECH_KEY='your-key'
export AZURE_SPEECH_REGION='your-resource-region'
export ELEVENLABS_API_KEY='your-key'
export ELEVENLABS_VOICE_ID='your-voice-id'
python cloud_tts_smoke.py
```

To use only one service:

```bash
python cloud_tts_smoke.py --provider azure
python cloud_tts_smoke.py --provider elevenlabs
```

Azure also supports excited, terrified, and shouting:

```bash
python cloud_tts_smoke.py --provider azure \
  --styles excited terrified shouting \
  --output outputs/azure-extra-smoke
```

Use `--styles neutral angry happy sad excited terrified shouting` for all seven.
These extra styles are currently exposed for Azure only; the default remains four styles.

Generate all seven with Guy, then classify them using the existing emotion2vec environment:

```bash
python cloud_tts_smoke.py --provider azure --azure-voice en-US-GuyNeural \
  --styles neutral angry happy sad excited terrified shouting \
  --output outputs/azure-guy-smoke && \
.venv-emotion/bin/python -u classify_cloud_smoke.py --path outputs/azure-guy-smoke
```

The classifier prints each requested style, predicted emotion, and top score.
Full label scores are saved to `results-emotion2vec.jsonl` in the smoke folder.
Emotion2vec does not have excited, terrified, or shouting labels: these are
classified into its own label set, without treating them as exact style matches.
Rerunning classification replaces that results file.

For 50 existing GSM8K questions in Jenny's angry, happy, and sad styles (150 clips):

```bash
python cloud_tts_smoke.py --provider azure --azure-voice en-US-JennyNeural \
  --input-jsonl outputs/text-prompt-prefix300/questions.jsonl --num-examples 50 \
  --styles angry happy sad --output outputs/azure-jenny50 && \
.venv-emotion/bin/python -u classify_cloud_smoke.py --path outputs/azure-jenny50
```

This uses the first 50 questions from the saved text experiment question file.
Alternatively supply another JSONL file with `question` and optional `question_id`/`id`, `answer`.
The classifier finishes with a requested-style by predicted-emotion count table and match rates.
Show the saved table again without loading the model:

```bash
python classify_cloud_smoke.py --path outputs/azure-jenny50 --summary-only
```

Compare GSM8K solving accuracy against the older cropped Ryan recordings, matching
question ID and requested TTS style (angry versus angry, happy versus happy):

```bash
python compare_cloud_accuracy.py --path outputs/azure-jenny50 --prepare-only
.venv-diffa/bin/python -u compare_cloud_accuracy.py --path outputs/azure-jenny50
```

This reuses `outputs/manual200-mixed/cropped/results-reasoning-v2.jsonl` for
older answers and evaluates only the matched Azure clips, skipping audio over
30 seconds. It does not require a neutral partner or filter by emotion labels.
It uses the existing DIFFA loader, reasoning prompt, 512-token limit, and strict
scorer, saving full Azure responses to `results-matched-accuracy.jsonl`.
Completed Azure evaluations are skipped on rerun. Unmatched questions are excluded;
the matching counts print before loading the model. Both styles have their own
matched cohort. Voices and TTS systems differ, so this comparison cannot isolate
emotion strength alone. The new run uses seed 42; older results retain their
original decoding randomness. Rerun reports during or after evaluation with:

```bash
python compare_cloud_accuracy.py --path outputs/azure-jenny50 --summary-only
```

Evaluate every Azure angry/happy/sad clip without requiring an older partner:

```bash
.venv-diffa/bin/python -u compare_cloud_accuracy.py --path outputs/azure-jenny50 \
  --all-cloud --styles angry happy sad
```

This reuses completed Azure evaluations in the same results file. The final table
shows all Azure accuracy versus all older Ryan/repeat-0 accuracy per style, with
separate denominators. This is descriptive: the question sets differ and no
paired significance test is applied. The 30-second audio limit still applies.
Use the same options with `--summary-only` to show progress without loading DIFFA.

To compare against only older clips whose intended style was also predicted by
emotion2vec, add `--older-emotion-agreed`. For example, this uses saved answers
only and reports overall accuracy against that filtered older subset:

```bash
python compare_cloud_accuracy.py --path outputs/azure-jenny50 \
  --all-cloud --styles angry happy sad --older-emotion-agreed --summary-only
```

The labels come from `outputs/manual200-mixed/cropped/results-emotion2vec-evaluated.jsonl`.
Missing labels are excluded. This flag filters the older side only; Azure remains
unfiltered. Omit `--all-cloud` for the smaller matched-question comparison with
this same older subset.

## Jenny: 500 additional questions per style

```bash
bash run_azure_jenny500.sh
```

The script selects input rows 50–549 from the saved full GSM8K question file,
excluding the previous first 50. It generates 1,500 clips (angry/happy/sad with
Jenny, intensity 1.5), classifies them with emotion2vec, then evaluates GSM8K
accuracy. Each stage finishes before the next starts. Outputs are separate in
`outputs/azure-jenny500-extra`. Azure generation resumes saved files only when
the saved request plan matches; evaluation also skips completed clips.
Classification reruns from the beginning and is fast. The generation stage
requires the Azure environment variables in the shell running the script.

The final report includes within-Azure comparisons matched by question ID and
voice, both for all requested deliveries and for emotion2vec agreement on both
sides. Each comparison excludes missing/over-30-second partners individually.
These are exploratory comparisons with unadjusted McNemar p-values. The all-clip
analysis avoids selecting the primary cohort by classifier predictions.

Check accuracy while evaluation runs, or show the final results again:

```bash
python compare_cloud_accuracy.py --path outputs/azure-jenny500-extra \
  --all-cloud --styles angry happy sad --summary-only
```

The complete pipeline now follows the emotional clips with 500 neutral Jenny
counterparts (the same input rows 50–549). Neutral clips are saved separately in
`outputs/azure-jenny500-neutral`, bringing the total to 2,000 clips. The final
combined report gives overall accuracy and paired comparisons for all four
deliveries, first unfiltered (primary) and then requiring emotion2vec agreement
on both sides. Pair counts can differ because each clip must fit within 30 seconds.

If the earlier three-style pipeline is already running, let it finish and run
the neutral stage separately afterward:

```bash
bash run_azure_jenny_neutral500.sh
```

Reprint the combined report anytime, without loading any model:

```bash
python analyze_jenny_deliveries.py
```

## Second classifier and matched-only accuracy reports

Classify the generated Jenny audio with SenseVoice using the existing emotion
environment. This leaves the emotion2vec results intact:

```bash
.venv-emotion/bin/python -u classify_cloud_smoke.py --model sensevoice \
  --path outputs/azure-jenny500-extra && \
.venv-emotion/bin/python -u classify_cloud_smoke.py --model sensevoice \
  --path outputs/azure-jenny500-neutral && \
python analyze_jenny_deliveries.py
```

The report shows intended-style agreement counts/rates for each classifier,
then accuracy only on matched question/voice pairs: unfiltered, emotion2vec-agreed,
SenseVoice-agreed, and agreement from both classifiers. In the filtered analyses,
both recordings in each pair must receive their intended emotion label. Whole-group
accuracy tables are omitted. Missing predictions are shown and excluded from
filtered comparisons. SenseVoice does not supply class scores in this adapter,
so its per-clip score column is `n/a`.

The matched report also includes **at least one classifier agrees**: each recording
in the pair must match its intended style under emotion2vec OR SenseVoice. The
classifier matching the neutral clip can differ from the one matching the emotional
clip. This is a per-recording union, followed by matching question/voice pairs;
both recordings still need to pass. Reprinting the report uses saved labels and
answers only, without model loading or new generation.

## Next independent 500-question batch

```bash
bash run_azure_jenny_next500.sh
```

This selects input rows 550–1049, without overlap with either earlier batch,
and runs generation, both classifiers, and evaluation for neutral/angry/happy/sad.
The output is `outputs/azure-jenny-next500`. Optional extra styles are positional:

```bash
bash run_azure_jenny_next500.sh terrified shouting
```

That example generates 3,000 clips: 500 questions times six styles. Jenny also
supports hopeful, friendly, unfriendly, shouting, and whispering. Extra styles
have no exact emotion2vec/SenseVoice label in these adapters; the report shows
their predicted-label distributions and unfiltered matched accuracy, without
inventing an exact emotion match. Use the same style arguments on rerun so the
saved generation plan remains identical.

Inspect the new batch independently, retaining it as a replication of the first:

```bash
python analyze_jenny_deliveries.py --paths outputs/azure-jenny-next500 \
  --styles neutral angry happy sad terrified shouting
```

You can subsequently pool the new and previous batches with `--paths` listing
`outputs/azure-jenny-next500 outputs/azure-jenny500-extra outputs/azure-jenny500-neutral`.
Only the new batch includes any additional styles; the matched comparisons
automatically use the available question partners.

## Nightly Jenny evaluation, angry vectors, and steering

```bash
bash run_jenny_angry_night.sh
```

The script waits for a detected running `run_azure_jenny_next500.sh` (or its
standalone new-batch accuracy evaluator), then uses saved audio to finish
classification/evaluation. It never invokes cloud generation and needs no Azure
credentials. Missing audio stops the queue with an explanation. Complete saved
classifier results skip model loading; the accuracy evaluator resumes completed
clips. It next extracts angry-minus-neutral vectors
from the **first** Jenny batch, requiring emotion2vec to label the angry and neutral
clips as intended and both to fit within 30 seconds. Selection never uses reasoning
correctness. Input paths are `outputs/azure-jenny500-extra` and
`outputs/azure-jenny500-neutral`. All question IDs in these training folders are
excluded from steering, including questions that fail the extraction filter.

Vectors go to `outputs/jenny-angry-vectors/style-vectors.pt` in the existing
steering format, with acoustic, semantic, combined, and answer groups, all layers,
per-forward aggregates, and an equal-question fixed direction. Forward indices
are matched only up to the shorter generation in each pair. Completed pairs are
checkpointed every 25 pairs and on normal exit/Ctrl-C; force-killing can lose the
work since the last checkpoint. Keep space for the roughly 1 GiB vector checkpoint
plus a temporary replacement during saving. The original extractor and `steer.py`
are unchanged.

Steering selects **50** new-batch neutral-generated clips by seed 42, without
classifier or correctness filtering. It excludes long audio and all training
question IDs. Each question gets a fresh unsteered baseline and +1 fixed angry
steering at each layer 0–25, on acoustic and answer tokens separately. This is
53 conditions per question, 2,650 generations; layers are not steered simultaneously.
The fixed direction is added at every decoder forward. Results go to
`outputs/steer-jenny-angry-L0-25-n50`, with per-condition resume.

Choose a specific running pipeline PID and/or a different steering count:

```bash
bash run_jenny_angry_night.sh auto 200
# 0 for the question count means all eligible new neutrals, a much longer sweep:
bash run_jenny_angry_night.sh auto 0
# Explicit PID instead of automatic detection:
bash run_jenny_angry_night.sh 12345 50
```

Only the new-batch pipeline is detected automatically; other DIFFA jobs must
finish before starting this sequence. If multiple matching pipelines exist, the
queue stops and asks for an explicit PID. A stopped process remains pending until
it is resumed or terminated. Model processes run sequentially; no second DIFFA
instance is launched by the queue while its detected pipeline remains alive.

Preparation and GPU smoke options, in separate output folders:

```bash
python extract_jenny_angry.py --prepare-only
.venv-diffa/bin/python -u extract_jenny_angry.py --limit 1 \
  --output outputs/jenny-angry-vector-smoke
.venv-diffa/bin/python -u steer_jenny_angry.py --num-examples 1 --layers 4 \
  --vectors outputs/jenny-angry-vector-smoke/style-vectors.pt \
  --output outputs/steer-jenny-angry-smoke
```

To print interim steering results without loading a model:

```bash
python - <<'PY'
import json
from pathlib import Path
from steer import read_rows, summarize
p = Path("outputs/steer-jenny-angry-L0-25-n50")
config = json.loads((p / "run_config.json").read_text())
summarize(read_rows(p / "results.jsonl"), config["settings"]["conditions"])
PY
```

## Check spoken question content from saved transcripts

SenseVoice classification already saves its ASR transcript in `raw_text`, so we
can audit all three Jenny folders without loading another model or using the GPU:

```bash
.venv-diffa/bin/python verify_jenny_content.py --evaluated-only
```

The report compares each supplied question with its saved SenseVoice transcript,
using Whisper's English text normalizer for punctuation and number formats (for
example `twelve` versus `12`). Parenthesized words and numeric negatives are
preserved before normalization. It reports normalized exact text agreement,
ordered number-token agreement, and word error rate per requested delivery.
`--evaluated-only` restricts the audit to clips included in GSM8K evaluation;
omit it to check all generated clips. Use `--paths` to select individual folders.

All comparisons go to `outputs/jenny-content-check/results.jsonl`; text mismatches
go to `review.csv`, sorted with number mismatches first. The CSV contains the
expected text, transcript, edits, and original audio path for listening. This
does not modify, filter, or regenerate any recordings or evaluation results.

ASR disagreement is a review flag, not proof that the TTS changed the question.
Conversely, normalized agreement does not prove perfectly identical spoken
content. Fractions, units, proper names, and other meaning changes need listening
or an independent recognizer for confirmation. Missing transcripts are reported
separately and excluded from agreement denominators.

Reprint the saved summary without importing Transformers:

```bash
python verify_jenny_content.py --summary-only
```

To listen to the five worst flagged clips alongside their neutral counterparts:

```bash
python listen_jenny_content.py
```

Download the entire `outputs/jenny-content-listen` folder and open `index.html`.
It includes copied audio, expected questions, and SenseVoice transcripts. Original
audio is unchanged. `--limit` changes the number of flagged examples; neutral
partners are additional clips. This is a deliberately selected review sample,
not an estimate of the dataset's error rate.

Outputs go to `outputs/cloud-tts-smoke/`: Azure WAVs, ElevenLabs MP3s, a
`plan.jsonl` containing the exact requests without keys, and `listen.html`.
Download the folder and open `listen.html` to compare the samples. Rerunning
generates the selected samples again and replaces their files.

Azure uses Jenny's SSML styles, with `cheerful` for happy and intensity 1.5.
Neutral uses the default delivery without a style tag. ElevenLabs uses
`eleven_v3` with `[angry]`, `[happily]`, and `[sad]` audio tags; neutral has no tag.
Choose an expressive ElevenLabs voice: tag responsiveness depends on the voice.
Listen for wording changes or spoken tags before using clips in an experiment.

Optional overrides: `--text "..."`, `--styles angry sad`,
`--azure-intensity 2`, `--azure-voice en-US-JennyNeural`,
`--eleven-model eleven_v3`, `--output outputs/my-smoke`.
ElevenLabs receives a fixed seed (default 42), which is best-effort reproducibility.
`--prepare-only` writes the plan and listening page without credentials or API calls.

For a focused follow-up of L3-answer and L5-acoustic at strength +1:

```bash
bash run_jenny_top2.sh
```

Both settings use the same 200 additional new-batch neutrals, excluding all
questions in `outputs/steer-jenny-angry-L0-25-n50/run_config.json` and all vector
extraction questions. Selection uses seed 42 and never correctness. Settings run
sequentially in separate processes, each with its own fresh baseline: 800 total
generations. Results resume in `outputs/steer-jenny-angry-L3-answer-new200` and
`outputs/steer-jenny-angry-L5-acoustic-new200`. The underlying `steer.py` is unchanged.

Provider documentation:

For the outcome-selected six-question strength/combination diagnostic:

```bash
python steer_jenny_six.py --prepare-only
.venv-diffa/bin/python -u steer_jenny_six.py
```

This freezes exactly `00708`, `00736`, `00746`, `00781`, `00792`, and `00987`
from the original 50-question Jenny sweep. Conditions are a fresh baseline,
L5-acoustic at x2, L5-acoustic x1 plus L0-acoustic or L11-answer at x1 and x0.5,
and x2 individually for each other layer/group that recovered at least one of
the six in the original sweep. With the current results, this is 20 conditions
and 120 generations. Combination hooks run together during the same generation;
each uses its own layer's fixed direction. Seeds match the previous sweep.

Results resume per question/condition in `outputs/steer-jenny-angry-six-followup`.
The output includes full answers and per-intervention counters. This is a selected
diagnostic, not an unbiased test of general accuracy. No Azure calls or audio
generation are performed. Existing steering scripts are unchanged.

Reprint interim or completed six-question results without a model:

```bash
python steer_jenny_six.py --summary-only
```

For the next six-question diagnostic, combine L5 acoustic x1 + L11 answer x1 +
L3 answer x2 simultaneously, and separately test the remaining previously
successful layer/group settings at x0.5:

```bash
.venv-diffa/bin/python -u steer_jenny_six.py --plan lower-strength
```

This excludes L5 acoustic, L11 answer, and L3 answer from the individual x0.5
tests. With the current original sweep, this is 12 individual tests, one triple
combination, and a fresh baseline: 84 generations across the same six questions.
Results are separate in `outputs/steer-jenny-angry-six-half`, with resume.

```bash
python steer_jenny_six.py --plan lower-strength --summary-only
```

For the remaining angry-correct/neutral-incorrect candidates in the new Jenny
batch, excluding all 50 prior sweep questions and all extraction questions:

```bash
.venv-diffa/bin/python -u steer_jenny_recovery.py
```

The current saved answers give 80 candidates, six already in the sweep, leaving
74. Each remaining neutral recording gets a fresh baseline, L5-acoustic x1,
L3-answer x2, and L11-answer x1, tested separately: 296 generations. Classifier
labels are not used for this selection. Results resume in
`outputs/steer-jenny-angry-recovery74`. The frozen settings are evaluated by the
same diagnostic runtime as the six-question experiments; `steer.py` is unchanged.
The recovery table excludes questions that already succeed under the fresh
baseline from its recovery denominator. This is an outcome-selected recovery
diagnostic, not the later regression test or an overall accuracy estimate.

```bash
python steer_jenny_recovery.py --prepare-only
python steer_jenny_recovery.py --summary-only
```

Expand the same frozen recovery cohort with the remaining zero-regression
candidates from the filtered 44-question view, plus L11 answer x1.5:

```bash
.venv-diffa/bin/python -u steer_jenny_recovery_followup.py
```

New individual targets are L0-answer x1, L2-acoustic x1, L3-answer x1,
L8-answer x1, L9-answer x1, L12-acoustic x1, L14-answer x1, L23-acoustic x1,
and L11-answer x1.5. The runner copies completed baseline/L5-acoustic x1/
L3-answer x2/L11-answer x1 results from `outputs/steer-jenny-angry-recovery74`
into `outputs/steer-jenny-angry-recovery74-expanded`, leaving the originals
unchanged. With all original conditions complete, 666 new generations remain.
It verifies vector identity and seeds before reusing saved results and freezes
the same 74 recordings. The final report includes old and new conditions.

```bash
python steer_jenny_recovery_followup.py --prepare-only
python steer_jenny_recovery_followup.py --summary-only
```

Provider documentation:

- [Azure REST TTS](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech)
- [Azure SSML styles](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice)
- [ElevenLabs TTS API](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)
- [ElevenLabs audio tags](https://elevenlabs.io/blog/v3-audiotags)
