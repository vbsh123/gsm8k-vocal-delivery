#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs

exec 9>outputs/jenny-task-vectors.lock
echo "$(date -Is) Waiting for the existing steering queue lock."
flock 9
while true; do
  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
  if [[ -z "$gpu_pids" ]]; then break; fi
  echo "$(date -Is) Waiting for GPU PID(s): $gpu_pids"
  sleep 30
done

exec .venv-diffa/bin/python -u run_large_steering.py
