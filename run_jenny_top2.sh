#!/usr/bin/env bash
# Evaluate the two candidates on the same 200 additional questions.
set -euo pipefail
cd "$(dirname "$0")"

N="${1:-200}"
EXCLUDE=outputs/steer-jenny-angry-L0-25-n50/run_config.json

.venv-diffa/bin/python -u steer_jenny_angry.py \
  --num-examples "$N" --exclude-runs "$EXCLUDE" \
  --layers 3 --groups answer --strength 1 \
  --output "outputs/steer-jenny-angry-L3-answer-new${N}"

.venv-diffa/bin/python -u steer_jenny_angry.py \
  --num-examples "$N" --exclude-runs "$EXCLUDE" \
  --layers 5 --groups acoustic --strength 1 \
  --output "outputs/steer-jenny-angry-L5-acoustic-new${N}"
