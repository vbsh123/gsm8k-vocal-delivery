#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/kimi-multiarith-queue.lock
flock -n 9 || { echo 'The Kimi MultiArith queue is already running.'; exit 1; }
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
echo 'STAGE: Freeze 290/290 split; reuse or resume neutral/angry Azure audio (CPU, no model)'
"$DIFFA_PYTHON" -u run_kimi_multiarith.py --mode audio "$@"
echo 'STAGE: Wait for the current Kimi BBH queues'
flock outputs/kimi-bbh-full-steering-queue.lock true
flock outputs/kimi-bbh-followups-queue.lock true
flock outputs/kimi-bbh-explicit-reasoning-queue.lock true
mkdir -p outputs/bbh-three-explicit-reasoning/kimi
flock outputs/bbh-three-explicit-reasoning/kimi/queue.lock true
echo 'STAGE: Kimi collection on ALL 290 pairs, then neutral/angry/L17-answer-x1 on ALL other 290'
"$KIMI_PYTHON" -u run_kimi_multiarith.py --mode evaluate "$@"
echo 'STAGE: CPU upstream scoring and paired uncertainty; original answers retained'
"$DIFFA_PYTHON" -u run_kimi_multiarith.py --mode summary "$@"
