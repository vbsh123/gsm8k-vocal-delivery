#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
exec "${DIFFA_PYTHON:-.venv-diffa/bin/python}" -u diffa_gsm_final_answer.py "$@"
