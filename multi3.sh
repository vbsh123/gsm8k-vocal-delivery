#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-run}" in
  run|prepare|summary) exec bash run_multiarith_reasoning_ablation.sh --three-asks --mode "${1:-run}" ;;
  start)
    mkdir -p outputs
    log=outputs/multiarith-three-asks.log
    nohup bash run_multiarith_reasoning_ablation.sh --three-asks >> "$log" 2>&1 < /dev/null &
    echo "Started PID $!; log: $log"
    ;;
  *) echo "Usage: bash multi3.sh [start|run|prepare|summary]" >&2; exit 2 ;;
esac
