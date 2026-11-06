#!/usr/bin/env bash
# 500 new questions x three deliveries; then classify and evaluate sequentially.
set -euo pipefail
cd "$(dirname "$0")"

OUTPUT="${1:-outputs/azure-jenny500-extra}"
QUESTIONS="${2:-outputs/text-prompt-prefix300/questions.jsonl}"

python -u cloud_tts_smoke.py \
  --provider azure --azure-voice en-US-JennyNeural \
  --input-jsonl "$QUESTIONS" --start-index 50 --num-examples 500 \
  --styles angry happy sad --output "$OUTPUT" --resume

.venv-emotion/bin/python -u classify_cloud_smoke.py --path "$OUTPUT"

.venv-diffa/bin/python -u compare_cloud_accuracy.py \
  --path "$OUTPUT" --all-cloud --styles angry happy sad

bash run_azure_jenny_neutral500.sh "$OUTPUT" outputs/azure-jenny500-neutral "$QUESTIONS"
