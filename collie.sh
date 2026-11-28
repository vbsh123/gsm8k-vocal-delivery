#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
log=outputs/diffa-collie-spoken.log
case "${1:-start}" in
  start) mkdir -p outputs; nohup bash collie.sh run >> "$log" 2>&1 </dev/null & echo "PID $! · $log"; exit ;;
  log) exec tail -n 30 -F "$log" ;;
  summary|run) ;;
  *) echo "Use start, log or summary." >&2; exit 2 ;;
esac
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
python="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
if [[ "$1" == run ]] && ! "$python" -c 'import dill, rich, nltk' 2>/dev/null; then
  "$python" -m pip install dill rich nltk
fi
exec "$python" -u collie_spoken.py "$1"
