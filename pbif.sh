#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
if [[ ${1:-} == kimi ]]; then
  shift
  if [[ ${1:-} == spoken ]]; then shift; exec bash kimi_ifeval.sh pbif-spoken "$@"; fi
  if [[ ${1:-} == pack ]]; then exec "${DIFFA_PYTHON:-.venv-diffa/bin/python}" -u kimi_pbif.py pack; fi
  exec bash kimi_ifeval.sh pbif "$@"
fi
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
runner=("${DIFFA_PYTHON:-.venv-diffa/bin/python}" -u pbif_emotions.py)
extra=()
log=outputs/pbif-neutral-angry-rate0.log
if [[ ${1:-} == steer ]]; then
  shift
  runner=("${DIFFA_PYTHON:-.venv-diffa/bin/python}" -u pbif_steering.py)
  log=outputs/pbif-angry-x1.log
elif [[ ${1:-} == spoken ]]; then
  shift
  runner=("${DIFFA_PYTHON:-.venv-diffa/bin/python}" -u diffa_pbif_spoken.py)
  log=outputs/diffa-pbif-spoken-neutral-angry.log
elif [[ ${1:-} == emotions ]]; then
  shift
  extra=(--emotions)
  log=outputs/pbif-added-styles-rate0.log
fi
case "${1:-run}" in
  start)
    mkdir -p outputs
    nohup "${runner[@]}" run "${extra[@]}" >> "$log" 2>&1 < /dev/null &
    echo "PID $! · $log" ;;
  log) exec tail -n 30 -F "$log" ;;
  prepare|run|summary) exec "${runner[@]}" "${1:-run}" "${extra[@]}" ;;
  *) echo "Usage: bash pbif.sh [spoken|emotions|steer] [start|prepare|run|log|summary]" >&2; exit 2 ;;
esac
