#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mode="${1:-run}"
if (($#)); then shift; fi
cpu="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
script=phi4_bbh_arithmetic_audio.py
if [[ "$mode" == run ]]; then
  "$cpu" -u "$script" prepare "$@"
  "${PHI_PYTHON:-.venv-bbh-phi4/bin/python}" -u "$script" run "$@"
  exec "$cpu" -u "$script" summary "$@"
fi
exec "$cpu" -u "$script" "$mode" "$@"
