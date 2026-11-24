#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cpu="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
phi="${PHI_PYTHON:-.venv-bbh-phi4/bin/python}"
dataset="${1:?Supply gsm8k, bbh5 or multiarith.}"
shift
if [[ "${1:-}" == --mode ]]; then shift; fi
case "${1:-run}" in
  prepare|summary) exec "$cpu" -u phi4_bare_reverse.py --dataset "$dataset" --mode "$1" ;;
  run) ;;
  *) echo 'Use run, prepare or summary.' >&2; exit 2 ;;
esac
"$cpu" -u phi4_bare_reverse.py --dataset "$dataset" --mode prepare
"$phi" -u phi4_bare_reverse.py --dataset "$dataset" --mode run
exec "$cpu" -u phi4_bare_reverse.py --dataset "$dataset" --mode summary
