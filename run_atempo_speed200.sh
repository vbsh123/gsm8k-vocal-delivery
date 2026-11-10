#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
python_bin="${PYTHON_BIN:-.venv-diffa/bin/python}"

# Extend the source pool: the first 300 have fewer than 200 clips <=15 seconds.
"$python_bin" -u prepare_spoken_gsm.py
args=(--num-examples 200 --stretch-method atempo
      --rates 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0
      --output outputs/spoken-gsm-atempo200 --save-audio)

echo "Preparing all 200 questions at 16 rates on CPU before evaluation."
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
echo "Evaluating atempo audio with DIFFA: 200 questions x 16 rates = 3200 answers."
"$python_bin" -u evaluate_audio_speed.py "${args[@]}"
