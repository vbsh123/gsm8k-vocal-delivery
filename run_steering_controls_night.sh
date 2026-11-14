#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
echo 'STAGE: Rescore the completed adaptive answers; no repeat collection or generation'
"$PYTHON" -u rescore_audio_steering.py --task gsm8k \
  --results outputs/jenny-L31-adaptive/evaluation/second/results.jsonl \
            outputs/jenny-L31-adaptive/evaluation/third/results.jsonl \
  --conditions same-state-mean-L31-a1 adaptive-L31-a0.5 adaptive-L31-a1 \
  --output outputs/jenny-L31-adaptive/scoring-audit
echo 'STAGE: Evaluate norm-matched random and reverse direction controls against the original mean'
"$PYTHON" -u run_L31_direction_controls.py
