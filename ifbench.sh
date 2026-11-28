#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
model="${1:-}"
case "$model" in
  audio) label=kimi-ifbench-recordings; python="${DIFFA_PYTHON:-.venv-diffa/bin/python}" ;;
  diffa) label=diffa-ifbench-flipped; python="${DIFFA_PYTHON:-.venv-diffa/bin/python}" ;;
  kimi) label=kimi-ifbench-spoken; python="${KIMI_PYTHON:-/venv/kimi-audio/bin/python}" ;;
  *) echo 'Use bash ifbench.sh {audio|diffa|kimi} {start|log|summary}' >&2; exit 2 ;;
esac
action="${2:-start}"
log="outputs/$label.log"
case "$action" in
  start) mkdir -p outputs; nohup bash ifbench.sh "$model" run >> "$log" 2>&1 </dev/null & echo "PID $! · $log"; exit ;;
  log) exec tail -n 25 -F "$log" ;;
  run|summary) ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
if [[ -z "${HF_HOME:-}" && -d /workspace/.hf_home/hub ]]; then export HF_HOME=/workspace/.hf_home; fi
if [[ -n "${HF_HOME:-}" ]]; then export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"; fi
if [[ "$model" == kimi && ! -x "$python" && -z "${KIMI_PYTHON:-}" ]]; then
  python="$(conda run -n kimi-audio python -c 'import sys; print(sys.executable)')"
fi
if [[ "$model" != audio && "$action" == run ]] && ! "$python" -c 'import absl, emoji, immutabledict, langdetect, nltk, syllapy' 2>/dev/null; then
  "$python" -m pip install absl-py emoji immutabledict langdetect nltk syllapy
fi
exec "$python" -u ifbench_placement.py "$model" "$action"
