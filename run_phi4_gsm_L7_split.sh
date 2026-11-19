#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
layer="${1:-7}"
[[ "$layer" =~ ^[0-9]+$ ]] && (( layer >= 0 && layer < 32 )) || { echo "Layer must be0..31"; exit 2; }
mkdir -p outputs
exec 8>outputs/phi4-gsm-L${layer}-wrapper.lock
flock -n 8 || { echo "L${layer} wrapper already active; stop the old serial L${layer} run first."; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
.venv-diffa/bin/python -u run_phi4_gsm_L7.py --mode prepare --layer "$layer"
echo 'STAGE: Wait for shared GPU access; then two Phi workers, one strength each.'
exec 9>outputs/jenny-task-vectors.lock
flock 9
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
    echo 'Waiting for active GPU work.'
    sleep 30
done
echo "STAGE: worker0=L${layer} answer -1; worker1=L${layer} answer -1.5. Same869 questions; same400-question vector; separate files."
.venv-bbh-phi4/bin/python -u run_phi4_gsm_L7.py --mode worker --layer "$layer" --worker-id 0 >outputs/phi4-gsm-L${layer}-worker0-minus1.log 2>&1 &
pid0=$!
.venv-bbh-phi4/bin/python -u run_phi4_gsm_L7.py --mode worker --layer "$layer" --worker-id 1 >outputs/phi4-gsm-L${layer}-worker1-minus1.5.log 2>&1 &
pid1=$!
status0=0 status1=0
wait "$pid0" || status0=$?
wait "$pid1" || status1=$?
echo "Workers finished: minus1=$status0; minus1.5=$status1"
((status0 == 0 && status1 == 0))
.venv-diffa/bin/python -u run_phi4_gsm_L7.py --mode summary --layer "$layer"
echo "Finished both L${layer} strengths; original and worker result files retained."
