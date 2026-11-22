#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python=.venv-bbh-phi4/bin/python
if [[ " $* " == *" --mode summary "* || " $* " == *" --mode prepare "* ]]; then
  python="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
fi
exec "$python" -u phi4_zip_recheck.py "$@"
