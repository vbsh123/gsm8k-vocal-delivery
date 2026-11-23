#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
role="${1:?Usage: bash run_emotion_block_night.sh old|phi|new [--mode prepare|run|summary] [--audio-root PATH]}"
shift
mode=run
extra=()
while (($#)); do
  case "$1" in
    --mode) mode="$2"; shift 2 ;;
    *) extra+=("$1"); shift ;;
  esac
done
case "$mode" in prepare|run|summary) ;; *) echo 'Mode must be prepare, run or summary.'; exit 2 ;; esac
case "$role" in
  old) datasets=(gsm8k multiarith) ;;
  phi) datasets=(svamp ifeval) ;;
  new) datasets=(bbh3 bbh5) ;;
  *) echo 'Role must be old, phi or new.'; exit 2 ;;
esac
mkdir -p outputs
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cpu="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
if [[ "$mode" != run ]]; then
  "$cpu" -u emotion_block_night.py --model diffa --datasets "${datasets[@]}" --mode "$mode" "${extra[@]}"
  if [[ "$role" == phi ]]; then
    "$cpu" -u emotion_block_night.py --model phi --datasets gsm8k --mode "$mode" "${extra[@]}"
  fi
  exit
fi
# Check every phase before starting GPU work. Different interpreters exit between models.
"$cpu" -u emotion_block_night.py --model diffa --datasets "${datasets[@]}" --mode prepare "${extra[@]}"
if [[ "$role" == phi ]]; then
  test -x .venv-bbh-phi4/bin/python
  "$cpu" -u emotion_block_night.py --model phi --datasets gsm8k --mode prepare "${extra[@]}"
fi
"$cpu" -u emotion_block_night.py --model diffa --datasets "${datasets[@]}" --mode run "${extra[@]}"
if [[ "$role" == phi ]]; then
  .venv-bbh-phi4/bin/python -u emotion_block_night.py --model phi --datasets gsm8k --mode run "${extra[@]}"
fi
