#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
layer="${1:-30}"
strength="${2:--1.5}"
mkdir -p outputs
exec 9>"outputs/phi4-gsm-extra-L${layer}-a${strength}.lock"
flock -n 9 || { echo 'This extra-strength run is already queued/running.'; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
echo 'Waiting for current Phi GSM/MultiArith follow-up queue; shared GPU lock follows.'
exec 8>outputs/phi4-reverse-followups-wrapper.lock
flock 8
.venv-bbh-phi4/bin/python -u phi4_reverse_followups.py --stage gsm-extra --layer "$layer" --strength "$strength"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u phi4_reverse_followups.py \
  --stage summary-gsm --layer "$layer" --output "outputs/phi4-gsm-reverse400-869-L${layer}-a${strength}"
