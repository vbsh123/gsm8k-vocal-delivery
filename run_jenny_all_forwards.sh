#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

.venv-diffa/bin/python -u extract_audio_vectors.py \
  --averaging all --methods mean \
  --output outputs/jenny-all-forwards

.venv-diffa/bin/python -u evaluate_audio_steering.py \
  --vectors-root outputs/jenny-all-forwards --methods mean --include-l11 \
  --output outputs/steer-jenny-all-forwards-second460
