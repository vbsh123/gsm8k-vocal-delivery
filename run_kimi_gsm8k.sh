#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/kimi-gsm8k-queue.lock
flock -n 9 || { echo 'Kimi GSM8K queue already running.'; exit 1; }
DIFFA_PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
KIMI_PYTHON="${KIMI_PYTHON:-/venv/kimi-audio/bin/python}"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
if [[ -z "${HF_HOME:-}" && -d /workspace/.hf_home/hub ]]; then
    export HF_HOME=/workspace/.hf_home
fi
if [[ -n "${HF_HOME:-}" ]]; then
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
fi
echo 'STAGE: Freeze 634/635 split; reuse/resume Azure audio on CPU'
"$DIFFA_PYTHON" -u run_kimi_gsm8k.py --mode audio "$@"
echo 'STAGE: Wait for Kimi MultiArith and existing BBH queues'
flock outputs/kimi-multiarith-queue.lock true
flock outputs/kimi-bbh-full-steering-queue.lock true
flock outputs/kimi-bbh-followups-queue.lock true
flock outputs/kimi-bbh-explicit-reasoning-queue.lock true
mkdir -p outputs/bbh-three-explicit-reasoning/kimi
flock outputs/bbh-three-explicit-reasoning/kimi/queue.lock true
echo 'STAGE: Collect on 634 pairs; evaluate neutral/angry/L17-answer-x1 on other 635'
"$KIMI_PYTHON" -u run_kimi_gsm8k.py --mode evaluate "$@"
"$DIFFA_PYTHON" -u run_kimi_gsm8k.py --mode summary "$@"
