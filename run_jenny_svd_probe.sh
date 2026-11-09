#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
.venv-diffa/bin/python -u extract_audio_vectors.py --methods mean svd1 svd_mean probe --limit 2 --output outputs/jenny-svd-probe-smoke
.venv-diffa/bin/python -u extract_audio_vectors.py --methods mean svd1 svd_mean probe --output outputs/jenny-svd-probe
.venv-diffa/bin/python -u evaluate_audio_steering.py
