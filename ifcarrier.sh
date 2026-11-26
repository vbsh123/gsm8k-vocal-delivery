#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
extra=()
log=outputs/ifeval-carrier.log
if [[ "${1:-}" == emotions ]]; then
  shift
  extra=(--emotions)
  log=outputs/ifeval-carrier-happy-sad.log
fi
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
if [[ "${1:-}" == phi ]]; then
  shift
  log=outputs/phi4-ifeval-written-carrier.log
  phi="${PHI_PYTHON:-.venv-bbh-phi4/bin/python}"
  scorer="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
  case "${1:-run}" in
    start)
      nohup bash ifcarrier.sh phi run >> "$log" 2>&1 < /dev/null &
      echo "Started PID $!; log: $log"
      exit ;;
    prepare) exec "$phi" -u phi4_ifeval_flip.py prepare ;;
    run) "$phi" -u phi4_ifeval_flip.py run ;;
    summary) ;;
    *) echo "Usage: bash ifcarrier.sh phi [prepare|run|start|summary]" >&2; exit 2 ;;
  esac
  exec "$scorer" -u phi4_ifeval_flip.py summary
fi
runner=("${DIFFA_PYTHON:-.venv-diffa/bin/python}" -u ifeval_carrier.py "${extra[@]}")
if [[ "${1:-run}" == start ]]; then
  nohup "${runner[@]}" --mode run >> "$log" 2>&1 < /dev/null &
  echo "Started PID $!; log: $log"
else
  exec "${runner[@]}" --mode "${1:-run}"
fi
