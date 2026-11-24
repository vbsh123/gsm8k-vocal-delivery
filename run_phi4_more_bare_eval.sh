#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mode=run
extra=()
while (($#)); do
  case "$1" in
    --mode) mode="${2:?--mode requires prepare, run or summary}"; shift 2 ;;
    *) extra+=("$1"); shift ;;
  esac
done
cpu="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
case "$mode" in
  prepare|summary) exec "$cpu" -u phi4_more_bare_eval.py --mode "$mode" "${extra[@]}" ;;
  run) ;;
  *) echo 'Mode must be prepare, run or summary.' >&2; exit 2 ;;
esac
phi="${PHI_PYTHON:-.venv-bbh-phi4/bin/python}"
test -x "$phi" || { echo "Missing Phi environment: $phi" >&2; exit 1; }
"$cpu" -u phi4_more_bare_eval.py --mode prepare "${extra[@]}"
exec "$phi" -u phi4_more_bare_eval.py --mode run "${extra[@]}"
