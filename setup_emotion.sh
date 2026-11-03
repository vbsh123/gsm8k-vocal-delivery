#!/usr/bin/env bash
# Keep emotion model dependencies separate from DIFFA and TTS.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10–3.12 (PYTHON_BIN=python3.12 if needed)"'
"$PYTHON_BIN" -m venv .venv-emotion
.venv-emotion/bin/python -m pip install --upgrade pip setuptools wheel
.venv-emotion/bin/python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv-emotion/bin/python -m pip install -r requirements-emotion.txt
.venv-emotion/bin/python -m pip check
printf '\nReady: .venv-emotion/bin/python emotion2vec_agreement.py\n'
