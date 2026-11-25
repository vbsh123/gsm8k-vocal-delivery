#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
extra=()
log=outputs/multiarith-three-asks.log
if [[ "${1:-}" == emotions ]]; then
  shift
  extra=(--happy-sad)
  log=outputs/multiarith-three-asks-happy-sad.log
fi
case "${1:-run}" in
  run|prepare|summary) exec bash run_multiarith_reasoning_ablation.sh --three-asks "${extra[@]}" --mode "${1:-run}" ;;
  start)
    mkdir -p outputs
    nohup bash run_multiarith_reasoning_ablation.sh --three-asks "${extra[@]}" >> "$log" 2>&1 < /dev/null &
    echo "Started PID $!; log: $log"
    ;;
  *) echo "Usage: bash multi3.sh [emotions] [start|run|prepare|summary]" >&2; exit 2 ;;
esac
