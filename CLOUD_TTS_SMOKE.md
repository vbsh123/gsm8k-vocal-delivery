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
