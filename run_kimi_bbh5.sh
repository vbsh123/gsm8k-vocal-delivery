#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
for arg in "$@"; do
    case "$arg" in
        --output|--output=*|--explicit-reasoning|--mode|--mode=*)
            echo 'This queue runs both prompts with separate fixed output folders; pass source/model options only.'
            exit 2 ;;
    esac
done
exec 9>outputs/kimi-bbh5-queue.lock
flock -n 9 || { echo 'Kimi BBH-five queue already running.'; exit 1; }
DIFFA_PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
KIMI_PYTHON="${KIMI_PYTHON:-/venv/kimi-audio/bin/python}"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
if [[ -z "${HF_HOME:-}" && -d /workspace/.hf_home/hub ]]; then export HF_HOME=/workspace/.hf_home; fi
if [[ -n "${HF_HOME:-}" ]]; then
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
fi
echo 'STAGE: Prepare all 250 neutral/angry pairs; freeze 125/125 split'
"$DIFFA_PYTHON" -u run_kimi_bbh5.py --mode audio "$@"
"$DIFFA_PYTHON" -u run_kimi_bbh5.py --mode prepare --explicit-reasoning "$@"
echo 'STAGE: Wait for current Kimi queues'
flock outputs/kimi-multiarith-queue.lock true
flock outputs/kimi-gsm8k-queue.lock true
flock outputs/kimi-bbh-full-steering-queue.lock true
flock outputs/kimi-bbh-followups-queue.lock true
flock outputs/kimi-bbh-explicit-reasoning-queue.lock true
mkdir -p outputs/bbh-three-explicit-reasoning/kimi
flock outputs/bbh-three-explicit-reasoning/kimi/queue.lock true
echo 'STAGE: All-250 delivery evaluation; collect 125 and steer other 125 at L17 answer x1'
"$KIMI_PYTHON" -u run_kimi_bbh5.py --mode evaluate "$@"
"$DIFFA_PYTHON" -u run_kimi_bbh5.py --mode summary "$@"
echo 'STAGE: NEW explicit-reasoning prompt; own 125-pair vector, same 125 test questions'
"$KIMI_PYTHON" -u run_kimi_bbh5.py --mode evaluate --explicit-reasoning "$@"
"$DIFFA_PYTHON" -u run_kimi_bbh5.py --mode summary --explicit-reasoning "$@"
