#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
model=${1:-}
placement=${2:-}
action=${3:-prepare}
case "$model/$placement" in
  diffa/spoken|diffa/flipped|kimi/spoken|kimi/flipped) ;;
  *) echo "Usage: bash pbif_full.sh {diffa|kimi} {spoken|flipped} [prepare|audio|start|run|log|summary]" >&2; exit 2 ;;
esac
log="outputs/$model-pbif-full-$placement.log"
case "$action" in
  start) mkdir -p outputs; nohup bash pbif_full.sh "$model" "$placement" run >> "$log" 2>&1 </dev/null & echo "PID $! · $log"; exit ;;
  log) exec tail -n 30 -F "$log" ;;
  prepare|audio|run|summary) ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
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
if [[ "$action" == run ]]; then
  "$scorer" -u pbif_full.py "$model" "$placement" build
  generator="$scorer"
  if [[ "$model" == kimi ]]; then generator="$native"; fi
  "$generator" -u pbif_full.py "$model" "$placement" generate
  action=summary
fi
exec "$scorer" -u pbif_full.py "$model" "$placement" "$action"
