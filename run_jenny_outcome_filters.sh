#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

exec 9>outputs/jenny-task-vectors.lock
echo "$(date -Is) Waiting for the existing task-vector queue to release the GPU."
flock 9
gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
if [[ -n "$gpu_pids" ]]; then
  echo "GPU is still used by PID(s): $gpu_pids. Existing work was left untouched." >&2
  exit 1
fi

for averaging in matched all; do
  if [[ "$averaging" == matched ]]; then
    collection=outputs/jenny-svd-probe
  else
    collection=outputs/jenny-all-forwards
  fi
  for selection in exclude_losses angry_correct; do
    .venv-diffa/bin/python -u extract_benefit_vectors.py \
      --collection "$collection" --filter "$selection" \
      --output "outputs/jenny-outcome-filters/$averaging"
  done
done

for averaging in matched all; do
  echo "$(date -Is) Evaluating $averaging: exclude losses vs angry correct only."
  .venv-diffa/bin/python -u evaluate_audio_steering.py \
    --vectors-root "outputs/jenny-outcome-filters/$averaging" \
    --methods exclude_losses_mean angry_correct_mean --include-l11 \
    --output "outputs/steer-jenny-outcome-filters/$averaging"
done
echo "$(date -Is) Both outcome-filter experiments finished."
