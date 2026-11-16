#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/kimi-bbh-followups-queue.lock
flock -n 9 || { echo 'A Kimi BBH follow-up queue is already running.'; exit 1; }
DIFFA_PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
KIMI_PYTHON="${KIMI_PYTHON:-/venv/kimi-audio/bin/python}"
export DIFFA_PYTHON KIMI_PYTHON
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
if [[ -z "${HF_HOME:-}" && -d /workspace/.hf_home/hub ]]; then
    export HF_HOME=/workspace/.hf_home
fi
if [[ -n "${HF_HOME:-}" ]]; then
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
fi
echo 'STAGE: Wait for the full recovery sweep; freeze layer choices before regression generation'
"$DIFFA_PYTHON" -u kimi_bbh_regressions.py --mode prepare --wait "$@"
echo 'STAGE: Existing vectors on regression neutral recordings; wait for GPU; no recollection'
"$KIMI_PYTHON" -u kimi_bbh_regressions.py --mode evaluate "$@"
"$DIFFA_PYTHON" -u kimi_bbh_regressions.py --mode summary "$@"
echo 'STAGE: All 250 neutral/angry pairs with the explicit reasoning prompt; saved progress resumes'
bash run_kimi_bbh_explicit_reasoning.sh
echo 'STAGE: Compare prompts, answer correctness, valid choices, lengths and delivery-by-prompt interaction'
"$DIFFA_PYTHON" -u analyze_kimi_reasoning_prompts.py
echo 'Finished Kimi BBH follow-ups.'
