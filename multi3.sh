#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
extra=(--three-asks)
log=outputs/multiarith-three-asks.log
if [[ "${1:-}" == emotions ]]; then
  shift
  extra+=(--happy-sad)
  log=outputs/multiarith-three-asks-happy-sad.log
elif [[ "${1:-}" == order-only ]]; then
  shift
  extra=(--order-only)
  log=outputs/multiarith-order-only270.log
elif [[ "${1:-}" == ready-format ]]; then
  shift
  extra=(--ready-format)
  log=outputs/multiarith-ready-format290.log
elif [[ "${1:-}" == sequenced ]]; then
  shift
  extra=(--sequenced)
  log=outputs/multiarith-sequenced20.log
elif [[ "${1:-}" == sequenced-rest ]]; then
  shift
  extra=(--sequenced-rest)
  log=outputs/multiarith-sequenced-rest270.log
fi
if [[ "${1:-}" == vectors ]]; then
  shift
  mkdir -p outputs
  unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
  export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
  export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
  worker=("${DIFFA_PYTHON:-.venv-diffa/bin/python}" -u multi3_prompt_vectors.py)
  case "${1:-run}" in
    start)
      nohup "${worker[@]}" run >> outputs/multiarith-prompt-vectors270.log 2>&1 < /dev/null &
      echo "Started PID $!; log: outputs/multiarith-prompt-vectors270.log"
      exit ;;
    run|summary) exec "${worker[@]}" "${1:-run}" ;;
    *) echo 'Usage: bash multi3.sh vectors [start|run|summary]' >&2; exit 2 ;;
  esac
fi
case "${1:-run}" in
  run|prepare|summary) exec bash run_multiarith_reasoning_ablation.sh "${extra[@]}" --mode "${1:-run}" ;;
  start)
    mkdir -p outputs
    nohup bash run_multiarith_reasoning_ablation.sh "${extra[@]}" >> "$log" 2>&1 < /dev/null &
    echo "Started PID $!; log: $log"
    ;;
  *) echo "Usage: bash multi3.sh [emotions|order-only|ready-format|sequenced|sequenced-rest|vectors] [start|run|prepare|summary]" >&2; exit 2 ;;
esac
