#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/kimi-bbh-explicit-reasoning-queue.lock
flock -n 9 || { echo 'The explicit-reasoning queue is already running.'; exit 1; }
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
mkdir -p outputs/bbh-three-explicit-reasoning/kimi
echo 'Waiting for any existing explicit-reasoning worker in this output directory.'
flock outputs/bbh-three-explicit-reasoning/kimi/queue.lock true
"$KIMI_PYTHON" -u evaluate_bbh_audio_models.py --model kimi \
    --source outputs/azure-bbh-three-models-regenerated \
    --output outputs/bbh-three-explicit-reasoning --explicit-reasoning "$@"
"$DIFFA_PYTHON" -u evaluate_bbh_audio_models.py --model kimi --mode summary \
    --output outputs/bbh-three-explicit-reasoning
