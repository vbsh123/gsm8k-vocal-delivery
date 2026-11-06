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

Provider documentation:

- [Azure REST TTS](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech)
- [Azure SSML styles](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice)
- [ElevenLabs TTS API](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)
- [ElevenLabs audio tags](https://elevenlabs.io/blog/v3-audiotags)
