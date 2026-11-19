#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/step-gsm400-wrapper.lock
flock -n 9 || { echo 'Step GSM queue is already running.'; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
echo 'STAGE: Freeze400 existing GSM8K neutral/angry pairs; no new audio.'
"$python" -u evaluate_step_gsm.py --mode prepare "$@"
echo 'STAGE: Check existing DIFFA environment and download pinned Step text-path weights on CPU.'
"$python" -u prepare_step_audio.py
echo 'STAGE: Wait for DIFFA transfer queue and GPU; then Step neutral/angry evaluation.'
"$python" -u evaluate_step_gsm.py --mode evaluate "$@"
