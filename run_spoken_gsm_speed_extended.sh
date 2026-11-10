#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs

exec 9>outputs/jenny-task-vectors.lock
echo "$(date -Is) Waiting for the shared GPU queue."
flock 9
while true; do
  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
  if [[ -z "$gpu_pids" ]]; then break; fi
  echo "$(date -Is) Waiting for GPU PID(s): $gpu_pids"
  sleep 30
done

exec .venv-diffa/bin/python -u evaluate_audio_speed.py \
  --num-examples 100 \
  --rates 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 \
  --reuse-from outputs/spoken-gsm-speed100 \
  --output outputs/spoken-gsm-speed100-extended
