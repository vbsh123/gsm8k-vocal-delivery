#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
runner=("${DIFFA_PYTHON:-.venv-diffa/bin/python}" -u pbif_emotions.py)
case "${1:-run}" in
  start)
    mkdir -p outputs
    nohup "${runner[@]}" run >> outputs/pbif-neutral-angry-rate0.log 2>&1 < /dev/null &
    echo "PID $! · outputs/pbif-neutral-angry-rate0.log" ;;
  prepare|run|summary) exec "${runner[@]}" "${1:-run}" ;;
  *) echo "Usage: bash pbif.sh [start|prepare|run|summary]" >&2; exit 2 ;;
esac
