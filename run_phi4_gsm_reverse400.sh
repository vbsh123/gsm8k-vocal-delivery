#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/phi4-gsm-reverse400-400-wrapper.lock
flock -n 9 || { echo 'Phi reverse400/400 is already queued.'; exit 1; }
DIFFA_PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
"$DIFFA_PYTHON" -u steer_phi4_gsm_reverse.py --mode prepare "$@"
echo 'STAGE: Wait for existing model and instruction-control queues; no jobs stopped'
for file in phi4-answer-only-queue gsm-candidate-phi4 bbh-candidate-phi4 bbh5-candidate-phi4 \
  phi4-reasoning-emotion100-queue diffa-reasoning-emotion100-queue \
  phi4-instruction-controls100-wrapper phi4-instruction-controls100-format-please-wrapper \
  diffa-instruction-controls100-wrapper diffa-instruction-controls100-format-please-wrapper; do
    flock "outputs/$file.lock" true
done
.venv-bbh-phi4/bin/python -u steer_phi4_gsm_reverse.py --mode evaluate "$@"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$DIFFA_PYTHON" -u steer_phi4_gsm_reverse.py --mode summary "$@"
