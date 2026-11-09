#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv-diffa/bin/python -u -m archive.steering.extract_jenny_svd_probe --limit 2 --output outputs/jenny-svd-probe-smoke
.venv-diffa/bin/python -u -m archive.steering.extract_jenny_svd_probe
.venv-diffa/bin/python -u evaluate_audio_steering.py
