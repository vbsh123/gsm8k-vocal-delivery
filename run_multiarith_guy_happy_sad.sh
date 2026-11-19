#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 8>outputs/multiarith-guy-happy-sad-wrapper.lock
flock -n 8 || { echo 'Guy happy/sad queue is already running.'; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
echo 'STAGE: Generate/resume Guy happy/sad clips on CPU; reuse all original neutral/angry responses.'
.venv-diffa/bin/python -u prepare_multiarith_guy_emotions.py
echo 'STAGE: Wait for shared GPU; transfer keeps running. Evaluate only missing happy/sad answers.'
.venv-diffa/bin/python -u run_multiarith_second_voice.py --mode evaluate --output outputs/multiarith290-guy-four-emotions
echo 'STAGE: CPU scoring of all four emotions; exploratory paired tests, unadjusted.'
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u run_multiarith_second_voice.py --mode summary --output outputs/multiarith290-guy-four-emotions
echo 'Finished Guy four-emotion comparison.'
