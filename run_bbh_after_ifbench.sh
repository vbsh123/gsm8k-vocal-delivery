#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
echo 'STAGE: Wait for IFBench and IFEval to finish'
"$PYTHON" -u wait_for_experiments.py --kind ifeval
"$PYTHON" -u wait_for_experiments.py --kind ifbench
echo 'STAGE: Install/verify pinned local harness API'
bash setup_harness.sh
echo 'STAGE: Web of Lies then Tracking Shuffled Objects; 200 questions each; neutral/angry/happy; no steering'
"$PYTHON" -u run_bbh_reasoning200.py "$@"
