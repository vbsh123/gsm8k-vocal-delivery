#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/minicpm-happy-next400-wrapper.lock
flock -n 9 || { echo 'MiniCPM next400 happy run is already queued/running.'; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
options=(--model minicpm --start-index 400 --num-questions 400 --require-neutral)
echo 'Prepare next400 neutral/happy recordings; existing neutral audio reused.'
.venv-diffa/bin/python -u evaluate_gsm_happy400.py --mode audio "${options[@]}"
echo 'Wait for existing MiniCPM evaluations; no new neutral generations.'
for file in multi-extra-happy-parallel-wrapper gsm-happy400-wrapper phi-minicpm-gsm-parallel-wrapper \
  minicpm-gsm400-steering-wrapper; do flock "outputs/$file.lock" true; done
.venv-diffa/bin/python -u evaluate_gsm_happy400.py --mode prepare "${options[@]}"
.venv-bbh-minicpm/bin/python -u evaluate_gsm_happy400.py --mode evaluate --concurrent "${options[@]}"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u evaluate_gsm_happy400.py --mode summary "${options[@]}"
