#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
extra=()
log=outputs/ifeval-carrier.log
if [[ "${1:-}" == emotions ]]; then
  shift
  extra=(--emotions)
  log=outputs/ifeval-carrier-happy-sad.log
fi
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
runner=("${DIFFA_PYTHON:-.venv-diffa/bin/python}" -u ifeval_carrier.py "${extra[@]}")
if [[ "${1:-run}" == start ]]; then
  nohup "${runner[@]}" --mode run >> "$log" 2>&1 < /dev/null &
  echo "Started PID $!; log: $log"
else
  exec "${runner[@]}" --mode "${1:-run}"
fi
