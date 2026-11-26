#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
if [[ -z "${HF_HOME:-}" && -d /workspace/.hf_home/hub ]]; then export HF_HOME=/workspace/.hf_home; fi
if [[ -n "${2:-}" ]]; then export IFEVAL_AUDIO_DIR="$2"; fi
mode="${1:-run}"
log=outputs/kimi-spoken-ifeval508-happy.log
if [[ "$mode" == start ]]; then
  mkdir -p outputs
  nohup bash kimi_ifeval.sh run >> "$log" 2>&1 < /dev/null &
  echo "PID $! · $log"
  exit
fi
if [[ "$mode" == setup ]]; then
  bash setup_kimi_audio.sh
fi
native="${KIMI_PYTHON:-/venv/kimi-audio/bin/python}"
if [[ ! -x "$native" ]] && command -v conda >/dev/null; then
  native="$(conda run -n kimi-audio python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
fi
scorer="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
if [[ -z "${DIFFA_PYTHON:-}" && ! -x "$scorer" ]]; then scorer="$native"; fi
require_python() {
  if [[ ! -x "$1" ]]; then echo 'Run bash kimi_ifeval.sh setup on the new machine first.' >&2; exit 1; fi
}
case "$mode" in
  setup)
    require_python "$scorer"
    "$scorer" -m pip install absl-py langdetect immutabledict nltk==3.9.2
    exec "$scorer" -m nltk.downloader punkt_tab ;;
  summary)
    require_python "$scorer"
    exec "$scorer" -u kimi_spoken_ifeval.py summary ;;
  prepare|run)
    require_python "$scorer"
    "$scorer" -u kimi_spoken_ifeval.py prepare
    if [[ "$mode" == prepare ]]; then exit; fi
    require_python "$native"
    "$native" -u kimi_spoken_ifeval.py generate
    exec "$scorer" -u kimi_spoken_ifeval.py summary ;;
  *) echo 'Usage: bash kimi_ifeval.sh [setup|prepare|start|run|summary] [Jenny folder or ZIP]' >&2; exit 2 ;;
esac
