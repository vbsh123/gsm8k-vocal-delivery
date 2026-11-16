#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
"$PYTHON" -u wait_for_experiments.py --kind winning-inputs
"$PYTHON" -u run_winning_strengths.py "$@"
