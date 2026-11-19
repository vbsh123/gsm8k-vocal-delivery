#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 8>outputs/phi4-gsm-L7-wrapper.lock
flock -n 8 || { echo 'L7 GSM follow-up already queued/running.'; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo 'STAGE: Full869 held-out GSM questions; reuse original400 L7 vector, neutral baselines and compatible diagnostic answers.'
.venv-diffa/bin/python -u run_phi4_gsm_L7.py --mode prepare
echo 'STAGE: Wait for shared GPU, then evaluate L7 answer -1/-1.5. SVAMP/BBH are not launched by this runner.'
exec 9>outputs/jenny-task-vectors.lock
flock 9
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
    echo 'Waiting for active GPU work.'
    sleep 30
done
.venv-bbh-phi4/bin/python -u run_phi4_gsm_L7.py --mode run
.venv-diffa/bin/python -u run_phi4_gsm_L7.py --mode summary
echo 'Finished L7 GSM follow-up.'
