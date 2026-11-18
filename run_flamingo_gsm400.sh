#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/flamingo-gsm400-wrapper.lock
flock -n 9 || { echo 'Flamingo GSM400 is already queued/running.'; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export PIP_NO_CACHE_DIR=1
options=(--model flamingo --prompt original --start-index 0 --num-questions 400 --output outputs/gsm400-candidates)
concurrent=false
if [[ "${1:-}" == --concurrent || "${FLAMINGO_CONCURRENT:-0}" == 1 ]]; then
    concurrent=true
    options+=(--concurrent --gpu-budget-gib 30)
elif [[ $# -gt 0 ]]; then echo 'Usage: bash run_flamingo_gsm400.sh [--concurrent]'; exit 2; fi
echo 'STAGE: Validate existing full GSM audio and freeze400 neutral/angry pairs; no TTS.'
.venv-diffa/bin/python -u evaluate_gsm_candidates.py --mode prepare "${options[@]}"
echo 'STAGE: Check missing environment/weight disk before any installation.'
.venv-diffa/bin/python -u flamingo_disk_preflight.py
bash setup_bbh_candidate.sh flamingo
if $concurrent; then
    echo 'STAGE: Concurrent Flamingo; skip queue/shared GPU lock;30 GiB Torch cap, require32 GiB free.'
else
    echo 'STAGE: Wait for existing model queues; Flamingo runs alone on GPU.'
    for file in multi-extra-happy-parallel-wrapper minicpm-happy-next400-wrapper gsm-happy400-wrapper \
      phi-minicpm-gsm-parallel-wrapper minicpm-gsm400-steering-wrapper phi4-reverse-followups-wrapper \
      phi4-gsm-extra-L17-a-1.5 phi4-gsm-extra-L17-a-2 phi4-gsm-extra-L30-a-2; do
        flock "outputs/$file.lock" true
    done
fi
.venv-bbh-flamingo/bin/python -u evaluate_gsm_candidates.py --mode evaluate "${options[@]}"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u evaluate_gsm_candidates.py --mode summary "${options[@]}"
