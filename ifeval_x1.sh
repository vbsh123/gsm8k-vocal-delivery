#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
log=outputs/diffa-ifeval-spoken-x1.log
case "${1:-start}" in
  start) nohup bash ifeval_x1.sh run >> "$log" 2>&1 </dev/null & echo "PID $! · $log"; exit ;;
  log) exec tail -n 30 -F "$log" ;;
  prepare|summary|run) ;;
  *) echo "Usage: bash ifeval_x1.sh [start|log|prepare|summary]" >&2; exit 2 ;;
esac
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
exec "${DIFFA_PYTHON:-.venv-diffa/bin/python}" -u diffa_ifeval_x1.py "$1"
