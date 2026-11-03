#!/usr/bin/env bash
# Classify original Spoken GSM sequentially; DIFFA is opt-in.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
prepare_args=()
classify_args=()
count="${1:-300}"
if [[ "$count" != all ]]; then
  prepare_args=(--num-examples "$count")
  classify_args=(--limit "$count")
fi
if [[ $# -gt 2 || ( -n "${2:-}" && "${2:-}" != --include-diffa ) ]]; then
  printf 'Usage: bash run_spoken_classification.sh [300|COUNT|all] [--include-diffa]\n' >&2
  exit 1
fi
# This Python env provides pyarrow/soundfile; preparation loads no DIFFA model.
.venv-diffa/bin/python -u prepare_spoken_gsm.py "${prepare_args[@]}"
if [[ "${2:-}" == --include-diffa ]]; then
  .venv-diffa/bin/python -u classify_spoken_gsm.py --model diffa "${classify_args[@]}"
fi
for model in emotion2vec sensevoice speechbrain; do
  .venv-emotion/bin/python -u classify_spoken_gsm.py --model "$model" "${classify_args[@]}"
done
