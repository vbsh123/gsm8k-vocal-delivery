#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>outputs/jenny-task-vectors.lock
flock -n 9 || { echo "This follow-up queue is already running." >&2; exit 1; }

wait_pid="${1:-auto}"
if [[ "$wait_pid" == auto ]]; then
  mapfile -t workers < <(pgrep -f '^bash (\./)?run_jenny_all_forwards\.sh( |$)' || true)
  if (( ${#workers[@]} > 1 )); then
    echo "Multiple current launchers found: ${workers[*]}. Pass the intended PID." >&2
    exit 1
  fi
  wait_pid="${workers[0]:-0}"
fi
if [[ ! "$wait_pid" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash run_task_vectors_after_current.sh [auto|CURRENT_LAUNCHER_PID|0]" >&2
  exit 1
fi
if (( wait_pid > 1 )); then
  started="$(ps -p "$wait_pid" -o lstart= || true)"
  echo "$(date -Is) Waiting for current run PID $wait_pid."
  while [[ -n "$started" && "$(ps -p "$wait_pid" -o lstart= || true)" == "$started" && "$(ps -p "$wait_pid" -o stat= || true)" != Z* ]]; do
    sleep 20
  done
fi

gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
if [[ -n "$gpu_pids" ]]; then
  echo "GPU is still in use by PID(s): $gpu_pids. Pass the current pipeline's launcher PID." >&2
  exit 1
fi

python_bin=.venv-diffa/bin/python
train_available=1
declare -A trained=([matched]=0 [all]=0)
echo "$(date -Is) Testing the new gradient-training path on two first-set questions."
if ! "$python_bin" -u train_answer_vectors.py \
  --collection outputs/jenny-svd-probe \
  --initial-vectors outputs/jenny-svd-probe/mean/style-vectors.pt \
  --output outputs/jenny-task-vector-smoke --limit 2 --epochs 1; then
  train_available=0
  echo "Gradient smoke failed. Continuing with both benefit-filtered vectors and their evaluations."
fi

for averaging in matched all; do
  if [[ "$averaging" == matched ]]; then
    collection=outputs/jenny-svd-probe
  else
    collection=outputs/jenny-all-forwards
  fi
  output="outputs/jenny-task-vectors/$averaging"
  echo "$(date -Is) Preparing $averaging benefit-filtered and answer-trained vectors."
  "$python_bin" -u extract_benefit_vectors.py --collection "$collection" --output "$output"
  if (( train_available )); then
    if "$python_bin" -u train_answer_vectors.py \
      --collection "$collection" --initial-vectors "$collection/mean/style-vectors.pt" --output "$output"; then
      trained[$averaging]=1
    else
      echo "$averaging answer training failed; its benefit-filtered evaluation will still run."
    fi
  fi
done

for averaging in matched all; do
  echo "$(date -Is) Evaluating $averaging vectors at L11 answer x1, L12 acoustic x1, L16 answer x1.25."
  methods=(benefit_mean)
  if (( trained[$averaging] )); then methods+=(answer_trained); fi
  "$python_bin" -u evaluate_audio_steering.py \
    --vectors-root "outputs/jenny-task-vectors/$averaging" \
    --methods "${methods[@]}" --include-l11 \
    --output "outputs/steer-jenny-task-vectors/$averaging"
done
echo "$(date -Is) Evaluations finished. Answer-trained variants completed: matched=${trained[matched]}, all=${trained[all]} (1=yes)."
