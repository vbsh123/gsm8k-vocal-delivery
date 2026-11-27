#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
log=outputs/diffa-spoken-ifeval508-terrified.log
case "${1:-start}" in
  start) mkdir -p outputs; nohup bash spoken_terrified508.sh run >> "$log" 2>&1 </dev/null & echo "PID $! · $log"; exit ;;
  log) exec tail -n 30 -F "$log" ;;
  run|summary|prepare) ;;
  *) echo "Use start, log, prepare, run or summary." >&2; exit 2 ;;
esac
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
exec "${DIFFA_PYTHON:-.venv-diffa/bin/python}" -u spoken_terrified508.py "$1"
