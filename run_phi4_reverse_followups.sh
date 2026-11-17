#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -eq 0 ]]; then echo 'Usage: bash run_phi4_reverse_followups.sh 17 [30]  (or 30 alone)'; exit 2; fi
layers=()
for layer in "$@"; do
    case "$layer" in 17|30) ;; *) echo 'Only layers17 and30 are supported.'; exit 2 ;; esac
    for old in "${layers[@]}"; do [[ "$old" != "$layer" ]] || { echo 'Duplicate layer.'; exit 2; }; done
    layers+=("$layer")
done
mkdir -p outputs
exec 9>outputs/phi4-reverse-followups-wrapper.lock
flock -n 9 || { echo 'Phi reverse follow-up queue is already running.'; exit 1; }
DIFFA_PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
echo "Chosen layers: ${layers[*]}; GSM remaining469 first, MultiArith290/290 second."
echo 'STAGE: Wait for current400/400, model and instruction-control queues'
for file in phi4-gsm-reverse400-400-wrapper phi4-answer-only-queue gsm-candidate-phi4 \
  bbh-candidate-phi4 bbh5-candidate-phi4 phi4-reasoning-emotion100-queue diffa-reasoning-emotion100-queue \
  phi4-instruction-controls100-wrapper phi4-instruction-controls100-format-please-wrapper \
  diffa-instruction-controls100-wrapper diffa-instruction-controls100-format-please-wrapper; do
    flock "outputs/$file.lock" true
done
for layer in "${layers[@]}"; do
    echo "STAGE: GSM remaining469 /L${layer}; original vector and400 test answers retained"
    .venv-bbh-phi4/bin/python -u phi4_reverse_followups.py --stage gsm-remaining --layer "$layer"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$DIFFA_PYTHON" -u phi4_reverse_followups.py --stage summary-gsm --layer "$layer"
done
for layer in "${layers[@]}"; do
    echo "STAGE: MultiArith own290/290 /L${layer}; strengths -0.5 and -1"
    .venv-bbh-phi4/bin/python -u phi4_reverse_followups.py --stage multiarith --layer "$layer"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$DIFFA_PYTHON" -u phi4_reverse_followups.py --stage summary-multiarith --layer "$layer"
done
