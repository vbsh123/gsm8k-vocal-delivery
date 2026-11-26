#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
out=outputs/jenny-terrified-math
mkdir -p "$out"
case "${1:-run}" in
  start)
    nohup bash terrified_math.sh run >> "$out/run.log" 2>&1 < /dev/null &
    echo "PID $! · $out/run.log"
    ;;
  prepare|run)
    export AZURE_SPEECH_KEY="${AZURE_SPEECH_KEY:-${SPEECH_KEY:-}}"
    export AZURE_SPEECH_REGION="${AZURE_SPEECH_REGION:-${SPEECH_REGION:-}}"
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    exec 9>"$out/worker.lock"
    flock -n 9 || { echo "Jenny terrified math worker already active."; exit 1; }
    exec python3 -u jenny_terrified_math.py "${1:-run}"
    ;;
  summary) exec python3 -u jenny_terrified_math.py summary ;;
  *) echo "Usage: bash terrified_math.sh [start|run|summary|prepare]" >&2; exit 2 ;;
esac
