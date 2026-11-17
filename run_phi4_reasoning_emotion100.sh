#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/phi4-reasoning-emotion100-queue.lock
flock -n 9 || { echo 'This vector-collection queue is already running.'; exit 1; }
DIFFA_PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
echo 'STAGE: Freeze SAME 100 questions, three conditions; no outcome filtering'
"$DIFFA_PYTHON" -u collect_phi4_reasoning_emotion.py --mode prepare "$@"
echo 'STAGE: Wait for current Phi-4 queues; nothing is stopped'
for file in phi4-answer-only-queue gsm-candidate-phi4 bbh-candidate-phi4 bbh5-candidate-phi4; do flock "outputs/$file.lock" true; done
[[ -x .venv-bbh-phi4/bin/python ]] || { echo 'Existing Phi-4 environment required.'; exit 1; }
.venv-bbh-phi4/bin/python -u collect_phi4_reasoning_emotion.py --mode collect "$@"
"$DIFFA_PYTHON" -u collect_phi4_reasoning_emotion.py --mode summary "$@"
