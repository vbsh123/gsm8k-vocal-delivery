#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-}" in
  old) datasets=(gsm8k multiarith) ;;
  new) datasets=(bbh3 bbh5) ;;
  phi) datasets=(svamp ifeval) ;;
  *) echo "Usage: bash compare_blocks.sh old|new|phi" >&2; exit 2 ;;
esac

unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
status=0
for dataset in "${datasets[@]}"; do
  echo "Comparing $dataset angry-minus-neutral vectors"
  "${DIFFA_PYTHON:-.venv-diffa/bin/python}" -u compare_diffa_prompt_vectors.py "$dataset" --emotion angry \
    --bare "outputs/emotion-block-night/diffa/$dataset/angry/style-vectors.pt" || status=1
done
exit "$status"
