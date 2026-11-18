#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
for task in svamp bbh3 bbh5; do
    echo "STAGE: Phi $task own-task vector and full held-out evaluation; supplied layer/settings frozen."
    .venv-bbh-phi4/bin/python -u phi4_task_steering.py --mode run --task "$task" "$@"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u phi4_task_steering.py --mode summary --task "$task" "$@"
done
