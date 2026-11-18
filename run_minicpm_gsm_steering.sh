#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/minicpm-gsm400-steering-wrapper.lock
flock -n 9 || { echo 'MiniCPM GSM steering is already queued/running.'; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
.venv-diffa/bin/python -u steer_minicpm_gsm.py --mode prepare "$@"
.venv-bbh-minicpm/bin/python -u steer_minicpm_gsm.py --mode run "$@"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u steer_minicpm_gsm.py --mode summary "$@"
