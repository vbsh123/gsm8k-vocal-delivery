#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/minicpm-gsm100-diagnostics-wrapper.lock
flock -n 9 || { echo 'MiniCPM diagnostics are already running.'; exit 1; }
DIFFA_PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export PIP_NO_CACHE_DIR=1
echo 'Concurrent MiniCPM diagnostics: no shared GPU lock, no Phi/job queue waits.'
"$DIFFA_PYTHON" -u diagnose_minicpm_gsm100.py --mode prepare "$@"
"$DIFFA_PYTHON" -u diagnose_minicpm_gsm100.py --mode disk "$@"
bash setup_bbh_candidate.sh minicpm
.venv-bbh-minicpm/bin/python -u diagnose_minicpm_gsm100.py --mode collect "$@"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$DIFFA_PYTHON" -u diagnose_minicpm_gsm100.py --mode summary "$@"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$DIFFA_PYTHON" -u diagnose_minicpm_gsm100.py --mode score "$@"
