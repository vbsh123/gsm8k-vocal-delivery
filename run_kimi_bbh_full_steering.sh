#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/kimi-bbh-full-steering-queue.lock
flock -n 9 || { echo 'The full Kimi steering queue is already running.'; exit 1; }
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
echo 'STAGE: Wait for the current follow-up queue and explicit-reasoning experiment to finish'
flock outputs/kimi-bbh-followups-queue.lock true
flock outputs/kimi-bbh-explicit-reasoning-queue.lock true
mkdir -p outputs/bbh-three-explicit-reasoning/kimi
flock outputs/bbh-three-explicit-reasoning/kimi/queue.lock true
echo 'STAGE: Freeze ALL original questions and OLD prompt; reuse compatible completed steering answers'
"$DIFFA_PYTHON" -u kimi_bbh_full_steering.py --mode prepare "$@"
echo 'STAGE: L16, L17 and L18 ANSWER x1 separately; existing vectors; no recollection or base model'
"$KIMI_PYTHON" -u kimi_bbh_full_steering.py --mode evaluate "$@"
echo 'STAGE: Upstream BBH scoring; all / collection-unseen / collection-seen'
"$DIFFA_PYTHON" -u kimi_bbh_full_steering.py --mode summary "$@"
