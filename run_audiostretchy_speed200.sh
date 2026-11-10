#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
python_bin="${PYTHON_BIN:-.venv-diffa/bin/python}"

"$python_bin" -u prepare_spoken_gsm.py
args=(--num-examples 200 --stretch-method audiostretchy
      --rates 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0
      --output outputs/spoken-gsm-audiostretchy200 --save-audio)

echo "Preparing 200 questions x 16 rates with AudioStretchy on CPU."
"$python_bin" -u evaluate_audio_speed.py "${args[@]}" --prepare-only

exec 9>outputs/jenny-task-vectors.lock
echo "Audio ready. Waiting for the GPU queue lock."
flock 9
while true; do
  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
  if [[ -z "$gpu_pids" ]]; then break; fi
  echo "Waiting for GPU PID(s): $gpu_pids"
  sleep 30
done
echo "Evaluating AudioStretchy audio with DIFFA: 3200 answers, seed 42 each time."
"$python_bin" -u evaluate_audio_speed.py "${args[@]}"
