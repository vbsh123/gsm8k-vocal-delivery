#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
if [[ ! -f outputs/commonsenseqa-angry-mean/style-vectors.pt ]]; then
  echo 'Missing existing CommonsenseQA vector: outputs/commonsenseqa-angry-mean/style-vectors.pt' >&2
  exit 1
fi
echo 'STAGE: Generate fixed ARC-Challenge questions in neutral/angry/happy/sad (Azure; CPU only)'
"$PYTHON" -u generate_arc_azure.py
echo 'STAGE: Wait for GPU; evaluate all deliveries and the unchanged CommonsenseQA L31 vector'
"$PYTHON" -u evaluate_arc_audio.py \
  --vectors outputs/commonsenseqa-angry-mean/style-vectors.pt
echo 'STAGE: Extract ARC L31 mean from 250 pairs; evaluate on the other 250 neutral recordings'
"$PYTHON" -u steer_arc_mean.py
echo 'STAGE: Emotion2vec classification and agreed-input accuracy summaries (saved answers only)'
"$PYTHON" -u run_arc_emotion2vec.py
