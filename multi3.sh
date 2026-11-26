#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
extra=(--three-asks)
log=outputs/multiarith-three-asks.log
if [[ "${1:-}" == emotions ]]; then
  shift
  extra+=(--happy-sad)
  log=outputs/multiarith-three-asks-happy-sad.log
elif [[ "${1:-}" == ready-format ]]; then
  shift
  extra=(--ready-format)
  log=outputs/multiarith-ready-format290.log
elif [[ "${1:-}" == sequenced ]]; then
  shift
  extra=(--sequenced)
  log=outputs/multiarith-sequenced20.log
fi
case "${1:-run}" in
  run|prepare|summary) exec bash run_multiarith_reasoning_ablation.sh "${extra[@]}" --mode "${1:-run}" ;;
  start)
    mkdir -p outputs
    nohup bash run_multiarith_reasoning_ablation.sh "${extra[@]}" >> "$log" 2>&1 < /dev/null &
    echo "Started PID $!; log: $log"
    ;;
  *) echo "Usage: bash multi3.sh [emotions|ready-format|sequenced] [start|run|prepare|summary]" >&2; exit 2 ;;
esac
