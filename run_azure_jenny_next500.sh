#!/usr/bin/env bash
# Next 500 unseen rows: neutral/angry/happy/sad plus optional extra Azure styles.
set -euo pipefail
cd "$(dirname "$0")"

OUTPUT=outputs/azure-jenny-next500
QUESTIONS=outputs/text-prompt-prefix300/questions.jsonl
STYLES=(neutral angry happy sad "$@")

python -u cloud_tts_smoke.py \
  --provider azure --azure-voice en-US-JennyNeural \
  --input-jsonl "$QUESTIONS" --start-index 550 --num-examples 500 \
  --styles "${STYLES[@]}" --output "$OUTPUT" --resume

for CLASSIFIER in emotion2vec sensevoice; do
  .venv-emotion/bin/python -u classify_cloud_smoke.py \
    --path "$OUTPUT" --model "$CLASSIFIER"
done

.venv-diffa/bin/python -u compare_cloud_accuracy.py \
  --path "$OUTPUT" --all-cloud --styles "${STYLES[@]}"

python analyze_jenny_deliveries.py --paths "$OUTPUT" --styles "${STYLES[@]}"
