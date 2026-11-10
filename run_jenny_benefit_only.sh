#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

exec 9>outputs/jenny-task-vectors.lock
echo "$(date -Is) Waiting for the GPU queue lock; finishing benefit_mean only."
flock 9
for averaging in matched all; do
  .venv-diffa/bin/python -u evaluate_audio_steering.py \
    --vectors-root "outputs/jenny-task-vectors/$averaging" \
    --methods benefit_mean --include-l11 \
    --output "outputs/steer-jenny-task-vectors/$averaging"
done
echo "$(date -Is) Both benefit_mean evaluations finished."
