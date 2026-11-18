#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/phi4-all-layer-diagnostic-wrapper.lock
flock -n 9 || { echo 'Phi all-layer diagnostic is already queued/running.'; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u phi4_all_layer_diagnostic.py --mode prepare "$@"
echo 'Waiting ONLY for Phi GSM L17 -2; no shared GPU lock or Flamingo queue wait.'
flock outputs/phi4-gsm-extra-L17-a-2.lock true
.venv-bbh-phi4/bin/python -u phi4_all_layer_diagnostic.py --mode collect "$@"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u phi4_all_layer_diagnostic.py --mode select "$@"
.venv-bbh-phi4/bin/python -u phi4_all_layer_diagnostic.py --mode steer "$@"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u phi4_all_layer_diagnostic.py --mode summary "$@"
