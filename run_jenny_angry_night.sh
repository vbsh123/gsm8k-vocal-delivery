#!/usr/bin/env bash
# Sequential evaluation -> vector extraction -> held-out steering, one DIFFA at a time.
set -euo pipefail
cd "$(dirname "$0")"

WAIT_PID="${1:-auto}"
N_NEUTRALS="${2:-50}"
if [[ "$WAIT_PID" != auto && ! "$WAIT_PID" =~ ^[0-9]+$ ]] || [[ ! "$N_NEUTRALS" =~ ^[0-9]+$ ]]; then
  echo 'Usage: bash run_jenny_angry_night.sh [auto|RUNNING_PIPELINE_PID|0] [NEUTRALS_OR_0_FOR_ALL]' >&2
  exit 1
fi
if [[ "$WAIT_PID" == auto ]]; then
  mapfile -t ACTIVE_PIDS < <(pgrep -f '^([^ ]*/)?bash (\./)?run_azure_jenny_next500\.sh( |$)' || true)
  if (( ${#ACTIVE_PIDS[@]} == 0 )); then
    mapfile -t ACTIVE_PIDS < <(pgrep -f '(^|/)python([0-9.]+)?( -u)? compare_cloud_accuracy\.py .*--path outputs/azure-jenny-next500( |$)' || true)
  fi
  if (( ${#ACTIVE_PIDS[@]} > 1 )); then
    echo "Multiple Jenny jobs found (${ACTIVE_PIDS[*]}); pass the intended pipeline PID explicitly." >&2
    exit 1
  fi
  WAIT_PID="${ACTIVE_PIDS[0]:-0}"
fi
if (( WAIT_PID > 1 )); then
  PROCESS_ARGS="$(ps -p "$WAIT_PID" -o args= || true)"
  if [[ -n "$PROCESS_ARGS" && "$PROCESS_ARGS" != *run_azure_jenny_next500.sh* && "$PROCESS_ARGS" != *compare_cloud_accuracy.py* ]]; then
    echo "PID $WAIT_PID is not a recognized Jenny pipeline command: $PROCESS_ARGS" >&2
    exit 1
  fi
  echo "$(date -Is) Waiting for Jenny pipeline PID $WAIT_PID."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    PROCESS_STATE="$(ps -p "$WAIT_PID" -o stat= || true)"
    [[ -z "$PROCESS_STATE" || "$PROCESS_STATE" == Z* ]] && break
    sleep 15
  done
fi

echo "$(date -Is) Finishing new audio classification and evaluation."
bash run_azure_jenny_next500.sh terrified shouting

echo "$(date -Is) Extracting angry-minus-neutral from the first Jenny batch."
.venv-diffa/bin/python -u extract_jenny_angry.py

echo "$(date -Is) Steering held-out new Jenny neutrals, L0-25, acoustic and answer separately."
.venv-diffa/bin/python -u steer_jenny_angry.py \
  --num-examples "$N_NEUTRALS" --output "outputs/steer-jenny-angry-L0-25-n${N_NEUTRALS}"

echo "$(date -Is) Jenny evaluation, extraction, and steering finished."
