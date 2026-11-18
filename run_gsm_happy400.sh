#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/gsm-happy400-wrapper.lock
flock -n 9 || { echo 'Happy400 evaluation already queued/running.'; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
echo 'STAGE: Prepare400 neutral/happy Azure pairs; reuse compatible existing audio.'
.venv-diffa/bin/python -u evaluate_gsm_happy400.py --mode audio "$@"
echo 'STAGE: Wait for current MiniCPM and Phi queues; no jobs stopped.'
for file in phi-minicpm-gsm-parallel-wrapper minicpm-gsm400-steering-wrapper phi4-reverse-followups-wrapper; do
    flock "outputs/$file.lock" true
done
for model in minicpm phi4; do
    echo "STAGE: $model happy versus neutral;400 matched questions; no steering."
    .venv-bbh-"$model"/bin/python -u evaluate_gsm_happy400.py --mode evaluate --model "$model" "$@"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u evaluate_gsm_happy400.py --mode summary --model "$model" "$@"
done
