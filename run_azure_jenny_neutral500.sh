#!/usr/bin/env bash
# Neutral counterparts to the same 500 new questions; keep the emotion run intact.
set -euo pipefail
cd "$(dirname "$0")"

EMOTIONS="${1:-outputs/azure-jenny500-extra}"
NEUTRAL="${2:-outputs/azure-jenny500-neutral}"
QUESTIONS="${3:-outputs/text-prompt-prefix300/questions.jsonl}"

python -u cloud_tts_smoke.py \
  --provider azure --azure-voice en-US-JennyNeural \
  --input-jsonl "$QUESTIONS" --start-index 50 --num-examples 500 \
  --styles neutral --output "$NEUTRAL" --resume

.venv-emotion/bin/python -u classify_cloud_smoke.py --path "$NEUTRAL"

.venv-diffa/bin/python -u compare_cloud_accuracy.py \
  --path "$NEUTRAL" --all-cloud --styles neutral

python analyze_jenny_deliveries.py --paths "$EMOTIONS" "$NEUTRAL"
