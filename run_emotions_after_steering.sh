#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 5>outputs/emotions-after-steering.lock
flock 5
for folder in azure-jenny500-extra azure-jenny500-neutral azure-jenny-next500 azure-jenny-remaining-test; do
  if [[ ! -f "outputs/$folder/plan.jsonl" ]]; then
    echo "Missing outputs/$folder/plan.jsonl; copy the recording folder before launching." >&2
    exit 1
  fi
done
python -u wait_for_experiments.py --kind steering
echo "Steering processes exited. Starting final-batch emotion classification and DIFFA accuracy."
bash run_jenny_third_accuracy.sh
