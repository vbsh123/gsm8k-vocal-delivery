#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
"$PYTHON" -m pip install absl-py langdetect nltk==3.9.2 immutabledict
"$PYTHON" -m nltk.downloader punkt punkt_tab
"$PYTHON" -u run_spoken_ifeval.py "$@"
