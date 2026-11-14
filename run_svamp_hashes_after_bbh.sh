#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
echo 'STAGE: Wait for active BBH jobs in this repository'
"$PYTHON" -u wait_for_experiments.py --kind bbh
echo 'STAGE: SVAMP neutral/angry with #### answers; saved audio, no steering'
"$PYTHON" -u run_svamp500.py --mode evaluate --unsteered-only --answer-format hashes
