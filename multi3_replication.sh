#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in phi|diffa-flip) experiment="$1" ;; *) echo 'Usage: bash multi3_replication.sh {phi|diffa-flip} {start|run|summary}' >&2; exit 2 ;; esac
mode="${2:-run}"
mkdir -p outputs
log="outputs/multi3-${experiment}290.log"
if [[ "$mode" == start ]]; then
  nohup bash "$0" "$experiment" run >> "$log" 2>&1 < /dev/null &
  echo "PID $! · $log"
  exit
fi
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
score="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
if [[ "$mode" == run ]]; then
  runner="$score"
  if [[ "$experiment" == phi ]]; then runner="${PHI_PYTHON:-.venv-bbh-phi4/bin/python}"; fi
  "$runner" -u run_multi3_replication.py "$experiment" run
  mode=summary
fi
exec "$score" -u run_multi3_replication.py "$experiment" "$mode"
