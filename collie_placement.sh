#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
model="${1:-}"
case "$model" in
  diffa) label=diffa-collie-flipped ;;
  kimi) label=kimi-collie-spoken ;;
  *) echo 'Usage: bash collie_placement.sh {diffa|kimi} {pack|results|prepare|start|run|log|summary}' >&2; exit 2 ;;
esac
mode="${2:-start}"
log="outputs/$label.log"
case "$mode" in
  start) mkdir -p outputs; nohup bash collie_placement.sh "$model" run >> "$log" 2>&1 </dev/null & echo "PID $! · $log"; exit ;;
  log) exec tail -n 30 -F "$log" ;;
  pack|results|prepare|summary|run) ;;
  *) echo "Unknown action: $mode" >&2; exit 2 ;;
esac
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
if [[ -z "${HF_HOME:-}" && -d /workspace/.hf_home/hub ]]; then export HF_HOME=/workspace/.hf_home; fi
if [[ -n "${HF_HOME:-}" ]]; then export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"; fi
native="${KIMI_PYTHON:-/venv/kimi-audio/bin/python}"
if [[ "$model" == kimi && ! -x "$native" ]] && command -v conda >/dev/null; then
  native="$(conda run -n kimi-audio python -c 'import sys; print(sys.executable)')"
fi
scorer="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
if [[ "$model" == kimi && ! -x "$scorer" && -z "${DIFFA_PYTHON:-}" ]]; then scorer="$native"; fi
if [[ "$mode" != summary ]] && ! "$scorer" -c 'import dill, nltk, rich' 2>/dev/null; then
  "$scorer" -m pip install dill nltk rich
fi
if [[ "$mode" == run ]]; then
  "$scorer" -u collie_placement.py "$model" prepare
  generator="$scorer"
  if [[ "$model" == kimi ]]; then generator="$native"; fi
  "$generator" -u collie_placement.py "$model" generate
  mode=summary
fi
exec "$scorer" -u collie_placement.py "$model" "$mode"
