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
