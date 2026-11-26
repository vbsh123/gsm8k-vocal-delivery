#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
if [[ -z "${HF_HOME:-}" && -d /workspace/.hf_home/hub ]]; then export HF_HOME=/workspace/.hf_home; fi
prefix=()
flags=()
log=outputs/kimi-spoken-ifeval508-happy.log
if [[ ${1:-} == extra ]]; then
  shift
  prefix=(extra)
  flags=(--extra)
  log=outputs/kimi-spoken-ifeval508-sad-terrified.log
  if [[ -n ${3:-} ]]; then export IFEVAL_TERRIFIED_AUDIO="$3"; fi
elif [[ ${1:-} == collect ]]; then
  shift
  prefix=(collect)
  log=outputs/kimi-spoken-ifeval-answer-vectors.log
elif [[ ${1:-} == pbif ]]; then
  shift
  prefix=(pbif)
  log=outputs/kimi-pbif-five-styles.log
fi
if [[ -n "${2:-}" ]]; then export IFEVAL_AUDIO_DIR="$2"; fi
mode="${1:-run}"
if [[ "$mode" == start && ${prefix[0]:-} != pbif ]]; then
  mkdir -p outputs
  nohup bash kimi_ifeval.sh "${prefix[@]}" run >> "$log" 2>&1 < /dev/null &
  echo "PID $! · $log"
  exit
fi
if [[ "$mode" == log && ${prefix[0]:-} == collect ]]; then exec tail -n 30 -F "$log"; fi
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
if [[ ${prefix[0]:-} == collect ]]; then
  require_python "$native"
  exec "$native" -u kimi_ifeval_collect.py "$mode"
fi
if [[ ${prefix[0]:-} == pbif ]]; then
  require_python "$scorer"
  case "$mode" in
    prepare|summary) exec "$scorer" -u kimi_pbif.py "$mode" ;;
    start)
      require_python "$native"
      "$scorer" -u kimi_pbif.py prepare
      mkdir -p outputs
      nohup bash kimi_ifeval.sh pbif queued-run >> "$log" 2>&1 < /dev/null &
      echo "Queued PID $! · $log"
      exit ;;
    queued-run|run)
      require_python "$native"
      if [[ "$mode" == run ]]; then "$scorer" -u kimi_pbif.py prepare; fi
      "$native" -u kimi_pbif.py "$mode"
      exec "$scorer" -u kimi_pbif.py summary ;;
    *) echo 'Usage: bash pbif.sh kimi [prepare|start|run|summary]' >&2; exit 2 ;;
  esac
fi
case "$mode" in
  setup)
    require_python "$scorer"
    "$scorer" -m pip install absl-py langdetect immutabledict nltk==3.9.2
    exec "$scorer" -m nltk.downloader punkt_tab ;;
  summary)
    require_python "$scorer"
    exec "$scorer" -u kimi_spoken_ifeval.py summary "${flags[@]}" ;;
  prepare|run)
    require_python "$scorer"
    "$scorer" -u kimi_spoken_ifeval.py prepare "${flags[@]}"
    if [[ "$mode" == prepare ]]; then exit; fi
    require_python "$native"
    "$native" -u kimi_spoken_ifeval.py generate "${flags[@]}"
    exec "$scorer" -u kimi_spoken_ifeval.py summary "${flags[@]}" ;;
  *) echo 'Usage: bash kimi_ifeval.sh [extra] [setup|prepare|start|run|summary] [Jenny folder or ZIP] [terrified ZIP or folder]' >&2; exit 2 ;;
esac
