#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
runner=("${DIFFA_PYTHON:-.venv-diffa/bin/python}" -u pbif_emotions.py)
extra=()
log=outputs/pbif-neutral-angry-rate0.log
if [[ ${1:-} == emotions ]]; then
  shift
  extra=(--emotions)
  log=outputs/pbif-added-styles-rate0.log
fi
case "${1:-run}" in
  start)
    mkdir -p outputs
    nohup "${runner[@]}" run "${extra[@]}" >> "$log" 2>&1 < /dev/null &
    echo "PID $! · $log" ;;
  prepare|run|summary) exec "${runner[@]}" "${1:-run}" "${extra[@]}" ;;
  *) echo "Usage: bash pbif.sh [emotions] [start|prepare|run|summary]" >&2; exit 2 ;;
esac
