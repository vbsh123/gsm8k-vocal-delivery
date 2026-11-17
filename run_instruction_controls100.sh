#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
model="${1:-diffa}"
case "$model" in diffa|phi4) ;; *) echo 'Usage: bash run_instruction_controls100.sh diffa|phi4 [options]'; exit 2 ;; esac
if [[ $# -gt 0 ]]; then shift; fi
mkdir -p outputs
exec 9>"outputs/${model}-instruction-controls100-wrapper.lock"
flock -n 9 || { echo 'This instruction-control queue is already running.'; exit 1; }
DIFFA_PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
"$DIFFA_PYTHON" -u collect_instruction_controls.py --model "$model" --mode prepare "$@"
echo 'STAGE: Wait for existing same100, model and DIFFA sweep queues; no jobs stopped'
for file in diffa-reasoning-emotion100-queue phi4-reasoning-emotion100-queue phi4-answer-only-queue gsm-candidate-phi4 bbh-candidate-phi4 bbh5-candidate-phi4; do
    flock "outputs/$file.lock" true
done
for folder in outputs/winning-strengths outputs/winning-transfer; do
    mkdir -p "$folder"
    flock "$folder/queue.lock" true
done
if [[ "$model" == phi4 ]]; then
    .venv-bbh-phi4/bin/python -u collect_instruction_controls.py --model "$model" --mode collect "$@"
else
    "$DIFFA_PYTHON" -u collect_instruction_controls.py --model "$model" --mode collect "$@"
fi
