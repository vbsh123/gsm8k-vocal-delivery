#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
if [[ "${1:-run}" == start ]]; then
  nohup bash ifcarrier.sh run >> outputs/ifeval-carrier.log 2>&1 < /dev/null &
  echo "Started PID $!; log: outputs/ifeval-carrier.log"
  exit
fi
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "${DIFFA_PYTHON:-.venv-diffa/bin/python}" -u ifeval_carrier.py --mode "${1:-run}"
