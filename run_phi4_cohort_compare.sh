#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "${DIFFA_PYTHON:-.venv-diffa/bin/python}" -u phi4_compare_delivery_cohorts.py "$@"
