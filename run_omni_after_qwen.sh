#!/usr/bin/env bash
# Install/download now; wait for the chosen Qwen process; smoke then first500.
set -euo pipefail
cd "$(dirname "$0")"
WAIT_PID="${1:-auto}"
if [[ "$WAIT_PID" == auto ]]; then
    mapfile -t ACTIVE < <(pgrep -f '(^|/)python([0-9.]+)?( -u)? ([^ ]*/)?evaluate_qwen_omni\.py( |$)' || true)
    if (( ${#ACTIVE[@]} > 1 )); then
        echo "Multiple Qwen workers found (${ACTIVE[*]}). Pass the intended Python PID explicitly." >&2
        exit 1
    fi
    WAIT_PID="${ACTIVE[0]:-0}"
fi
[[ "$WAIT_PID" =~ ^[0-9]+$ ]] || { echo 'Usage: bash run_omni_after_qwen.sh [auto|QWEN_PYTHON_PID|0]'; exit 1; }
WAIT_START=""
if (( WAIT_PID > 1 )); then
    PROCESS_ARGS="$(ps -p "$WAIT_PID" -o args= || true)"
    [[ -z "$PROCESS_ARGS" || "$PROCESS_ARGS" == *evaluate_qwen_omni.py* ]] || { echo 'The supplied PID is not Qwen.'; exit 1; }
    WAIT_START="$(ps -p "$WAIT_PID" -o lstart= || true)"
fi
echo "$(date -Is) Selected Qwen PID $WAIT_PID. Preparing Omni environment and downloads now; model loading waits."
bash setup_omni_diffusion.sh
conda run --no-capture-output -n omni-diffusion python -u evaluate_omni_diffusion.py --prepare-only
echo "$(date -Is) Preflight finished. Waiting for Qwen PID $WAIT_PID to exit."
POLL=0
while (( WAIT_PID > 1 )) && kill -0 "$WAIT_PID" 2>/dev/null; do
    CURRENT_START="$(ps -p "$WAIT_PID" -o lstart= || true)"
    STATE="$(ps -p "$WAIT_PID" -o stat= || true)"
    [[ -z "$STATE" || "$STATE" == Z* || "$CURRENT_START" != "$WAIT_START" ]] && break
    if (( POLL % 6 == 0 )); then
        echo "$(date -Is) Still waiting: Qwen PID $WAIT_PID, state $STATE."
    fi
    sleep 10
    POLL=$((POLL + 1))
done
echo "$(date -Is) Qwen process ended. Starting Omni-Diffusion smoke (3 questions x 4 deliveries)."
conda run --no-capture-output -n omni-diffusion python -u evaluate_omni_diffusion.py --smoke-gate
echo "$(date -Is) Smoke format gate passed. Evaluating the first500 neutral/angry/happy/sad clips."
conda run --no-capture-output -n omni-diffusion python -u evaluate_omni_diffusion.py --num-questions 0
python analyze_jenny_deliveries.py \
    --paths outputs/azure-jenny500-extra outputs/azure-jenny500-neutral \
    --styles neutral angry happy sad --results-file results-omni-diffusion.jsonl
echo "$(date -Is) Omni-Diffusion follow-up finished."
