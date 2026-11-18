#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
echo 'Prepare SVAMP500/500 and BBH3/5 125/125. CPU/TTS only; no GPU waiting/lock or model inference.'
.venv-diffa/bin/python -u phi4_task_steering.py --mode prepare --task all "$@"
