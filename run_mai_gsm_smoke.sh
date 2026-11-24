#!/usr/bin/env bash
# Exactly one GSM8K test question, four deliveries; Azure TTS only, no GPU work.
set -euo pipefail
cd "$(dirname "$0")"
export AZURE_SPEECH_KEY="${AZURE_SPEECH_KEY:-${SPEECH_KEY:-}}"
export AZURE_SPEECH_REGION="${AZURE_SPEECH_REGION:-${SPEECH_REGION:-}}"
# MAI Olivia supports neutral/angry/happy/sad. Use MAI's "happy", not Jenny's "cheerful".
# https://learn.microsoft.com/en-us/azure/ai-services/speech-service/mai-voices
exec python3 cloud_tts_smoke.py --provider azure \
  --azure-voice 'en-US-Olivia:MAI-Voice-2.1' \
  --text "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for \$2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?" \
  --num-examples 1 --styles neutral angry happy sad \
  --output outputs/mai-gsm-one --resume "$@"
