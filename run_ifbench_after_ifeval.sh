#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
echo 'STAGE: Wait for current IFEval jobs'
"$PYTHON" -u wait_for_experiments.py --kind ifeval
echo 'STAGE: Prepare official IFBench verifier dependencies'
"$PYTHON" -m pip install absl-py langdetect nltk==3.9.2 immutabledict emoji syllapy
"$PYTHON" -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger_eng stopwords
echo 'STAGE: IFBench neutral/angry/happy; generate then evaluate; no steering'
"$PYTHON" -u run_spoken_ifbench.py "$@"
