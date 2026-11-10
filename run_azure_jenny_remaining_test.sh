#!/usr/bin/env bash
# Generate only; classification and evaluation wait until the GPU is available.
set -euo pipefail
cd "$(dirname "$0")"

QUESTIONS=outputs/text-prompt-prefix300/questions.jsonl
OUTPUT=outputs/azure-jenny-remaining-test
START=1050
COUNT="$(python - "$QUESTIONS" "$START" <<'PY'
import json
from pathlib import Path
import sys

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
start = int(sys.argv[2])
if len(rows) <= start:
    raise SystemExit('The question bank has no test questions after index 1049.')
for index, row in enumerate(rows):
    if row['question_id'] != f'gsm8k_test_{index:05d}':
        raise SystemExit('Expected the original ordered GSM8K test question bank.')
print(len(rows)-start)
PY
)"

echo "$COUNT remaining test questions x 4 deliveries = $((COUNT * 4)) Azure clips."
echo "Separate from both earlier Azure batches; older experiment overlap must be checked before validation."
python -u cloud_tts_smoke.py \
  --provider azure --azure-voice en-US-JennyNeural --azure-intensity 1.5 \
  --input-jsonl "$QUESTIONS" --start-index "$START" --num-examples "$COUNT" \
  --styles neutral angry happy sad --output "$OUTPUT" --resume

echo "Generation finished. Classification and evaluation were not started; the steering GPU is untouched."
