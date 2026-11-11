#!/usr/bin/env bash
# Whole-recording emotion labels and DIFFA accuracy; no TTS or vector extraction.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 7>outputs/jenny-third-accuracy.lock
flock 7
device="${1:-cuda:0}"

python -u run_azure_followups.py --stage classify \
  --paths outputs/azure-jenny-remaining-test \
  --styles neutral angry happy sad --device "$device"

python -u run_azure_followups.py --stage diffa \
  --paths outputs/azure-jenny-remaining-test \
  --styles neutral angry happy sad

echo "All three Azure cohorts combined, matched question and voice."
.venv-diffa/bin/python -u analyze_jenny_deliveries.py \
  --paths outputs/azure-jenny500-extra outputs/azure-jenny500-neutral \
          outputs/azure-jenny-next500 outputs/azure-jenny-remaining-test \
  --styles neutral angry happy sad --confidence-intervals
