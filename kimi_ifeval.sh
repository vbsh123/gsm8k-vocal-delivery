#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
if [[ -z "${HF_HOME:-}" && -d /workspace/.hf_home/hub ]]; then export HF_HOME=/workspace/.hf_home; fi
scorer="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
native="${KIMI_PYTHON:-/venv/kimi-audio/bin/python}"
audio="${IFEVAL_AUDIO_DIR:-outputs/azure-spoken-ifeval}"
case "${1:-run}" in
  start)
    mkdir -p outputs
    nohup bash kimi_ifeval.sh run >> outputs/kimi-spoken-ifeval508.log 2>&1 < /dev/null &
    echo "PID $! · outputs/kimi-spoken-ifeval508.log" ;;
  summary) exec "$scorer" -u kimi_spoken_ifeval.py summary ;;
  prepare|run)
    "$scorer" -u kimi_spoken_ifeval.py prepare --source "$audio"
    if [[ "${1:-run}" == prepare ]]; then exit; fi
    if [[ ! -x "$native" ]] && command -v conda >/dev/null; then
      native="$(conda run -n kimi-audio python -c 'import sys; print(sys.executable)')"
    fi
    if [[ ! -x "$native" ]]; then echo 'Set KIMI_PYTHON to the Kimi environment Python, or run bash setup_kimi_audio.sh.' >&2; exit 1; fi
    "$native" -u kimi_spoken_ifeval.py generate
    exec "$scorer" -u kimi_spoken_ifeval.py summary ;;
  *) echo 'Usage: bash kimi_ifeval.sh [prepare|start|run|summary]' >&2; exit 2 ;;
esac
