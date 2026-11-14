#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
mkdir -p outputs

echo 'STAGE: Rescore existing CommonsenseQA answers (CPU only)'
"$PYTHON" -u audit_commonsense_scoring.py
"$PYTHON" -u rescore_audio_steering.py --task commonsenseqa \
  --output outputs/commonsenseqa-independent-audit-v2

echo 'STAGE: Generate the 721 unused validation questions, neutral only (Azure, no GPU)'
"$PYTHON" -u generate_commonsense_azure.py \
  --start-index 500 --num-examples 721 --styles neutral \
  --output outputs/azure-commonsenseqa-remaining721

echo 'STAGE: Wait for GPU, then evaluate baseline and the unchanged L31 vector'
"$PYTHON" -u extend_commonsense_steering.py
echo 'STAGE: Finished; new-721 and combined-heldout rescoring reports saved'
