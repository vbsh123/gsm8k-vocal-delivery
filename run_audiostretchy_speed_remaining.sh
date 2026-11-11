#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 8>outputs/spoken-gsm-audiostretchy-remaining.lock
flock 8
python_bin="${PYTHON_BIN:-.venv-diffa/bin/python}"
pilot=outputs/spoken-gsm-audiostretchy200
if [[ ! -f "$pilot/run_config.json" ]]; then
  echo "Copy $pilot/run_config.json from the 200-question speed machine first; it identifies the questions to exclude." >&2
  exit 1
fi
"$python_bin" -u prepare_spoken_gsm.py
args=(--num-examples 0 --stretch-method audiostretchy --rates 1 1.2 1.3
      --exclude-run "$pilot" --output outputs/spoken-gsm-audiostretchy-remaining --save-audio)
"$python_bin" -u evaluate_audio_speed.py "${args[@]}" --prepare-only

exec 9>outputs/jenny-task-vectors.lock
echo "Waiting for the GPU queue lock."
flock 9
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
  echo "Waiting for active GPU work."
  sleep 30
done
"$python_bin" -u evaluate_audio_speed.py "${args[@]}"
