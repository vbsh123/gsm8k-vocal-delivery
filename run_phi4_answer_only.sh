#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/phi4-answer-only-queue.lock
flock -n 9 || { echo 'Phi-4 answer-only follow-up is already running.'; exit 1; }
DIFFA_PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
echo 'STAGE: Stop current Phi-4 GSM8K queue only; completed answers retained'
"$DIFFA_PYTHON" -u phi4_answer_only.py --mode stop-gsm "$@"
echo 'STAGE: Freeze exact completed neutral/angry pairs from saved GSM8K and BBH-three answers'
"$DIFFA_PYTHON" -u phi4_answer_only.py --mode prepare "$@"
echo 'STAGE: Wait for other already queued Phi-4 evaluations'
flock outputs/gsm-candidate-phi4.lock true
flock outputs/bbh-candidate-phi4.lock true
flock outputs/bbh5-candidate-phi4.lock true
[[ -x .venv-bbh-phi4/bin/python ]] || { echo 'Existing Phi-4 environment is required.'; exit 1; }
echo 'STAGE: Answer-only neutral/angry; same frozen questions, native decoder, seed42, 512-token budget'
.venv-bbh-phi4/bin/python -u phi4_answer_only.py --mode evaluate "$@"
"$DIFFA_PYTHON" -u phi4_answer_only.py --mode summary "$@"
