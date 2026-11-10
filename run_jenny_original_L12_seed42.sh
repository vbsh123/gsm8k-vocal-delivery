#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

exec 9>outputs/jenny-task-vectors.lock
echo "$(date -Is) Waiting for the GPU queue lock; original vector, L12 acoustic x1, seed42."
flock 9
gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
if [[ -n "$gpu_pids" ]]; then
  echo "GPU is still used by PID(s): $gpu_pids. Existing work was left untouched." >&2
  exit 1
fi

original_vector="$(.venv-diffa/bin/python - <<'PY'
import json
from pathlib import Path
config = json.loads(Path('outputs/steer-jenny-angry-complete/run_config.json').read_text())
print(config['settings']['vectors'])
PY
)"

.venv-diffa/bin/python -u evaluate_audio_steering.py \
  --vectors "$original_vector" --methods original_mean \
  --conditions L12-acoustic-a1 \
  --output outputs/steer-jenny-original-vector-L12-seed42
