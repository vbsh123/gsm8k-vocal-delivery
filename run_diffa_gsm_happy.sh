#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "${DIFFA_PYTHON:-.venv-diffa/bin/python}" -u diffa_gsm_emotion.py --emotion happy "$@"
