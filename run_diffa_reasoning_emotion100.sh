#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/diffa-reasoning-emotion100-queue.lock
flock -n 9 || { echo 'DIFFA direction comparison already queued.'; exit 1; }
DIFFA_PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
"$DIFFA_PYTHON" -u collect_diffa_reasoning_emotion.py --mode prepare "$@"
echo 'STAGE: Wait for existing DIFFA strength and transfer queues'
for folder in outputs/winning-strengths outputs/winning-transfer; do
    mkdir -p "$folder"
    flock "$folder/queue.lock" true
done
"$DIFFA_PYTHON" -u collect_diffa_reasoning_emotion.py --mode collect "$@"
"$DIFFA_PYTHON" -u collect_diffa_reasoning_emotion.py --mode summary "$@"
