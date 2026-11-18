#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
phase="${1:-collect}"
if [[ $# -gt 0 ]]; then shift; fi
case "$phase" in collect|test) ;; *) echo 'Usage: bash run_phi4_all_layers_split.sh collect|test [options]'; exit 2 ;; esac
mkdir -p outputs
exec 9>outputs/phi4-all-layer-diagnostic-wrapper.lock
flock -n 9 || { echo 'An all-layer wrapper is already running. Stop only the old sequential diagnostic before switching.'; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
cpu=.venv-diffa/bin/python
native=.venv-bbh-phi4/bin/python
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$cpu" -u phi4_all_layer_diagnostic.py --mode prepare "$@"
echo 'Waiting ONLY for Phi L17 -2; two independent Phi workers follow.'
flock outputs/phi4-gsm-extra-L17-a-2.lock true
run_pair() {
    local mode="$1"
    shift
    local status0=0 status1=0
    "$native" -u phi4_all_layer_diagnostic.py --mode "$mode" --workers 2 --worker-id 0 --gpu-budget-gib 18 "$@" \
      >"outputs/phi4-all-layers-${mode}-worker0.log" 2>&1 &
    local pid0=$!
    "$native" -u phi4_all_layer_diagnostic.py --mode "$mode" --workers 2 --worker-id 1 --gpu-budget-gib 18 "$@" \
      >"outputs/phi4-all-layers-${mode}-worker1.log" 2>&1 &
    local pid1=$!
    wait "$pid0" || status0=$?
    wait "$pid1" || status1=$?
    echo "$mode finished: worker0=$status0 worker1=$status1"
    (( status0 == 0 && status1 == 0 ))
}
if [[ "$phase" == collect ]]; then
    echo 'STAGE: Collect first200/last200 pairs separately; all32 layers in each generation.'
    run_pair collect-shard "$@"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$cpu" -u phi4_all_layer_diagnostic.py --mode merge-vectors "$@"
    echo 'Collection complete. Merged400 equally weighted pairs. STOPPING before test references/steering.'
    echo 'Later run: bash run_phi4_all_layers_split.sh test'
else
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$cpu" -u phi4_all_layer_diagnostic.py --mode merge-vectors "$@"
    echo 'STAGE: Split remaining869 questions into434/435; generate missing neutral/angry references only.'
    run_pair references-shard "$@"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$cpu" -u phi4_all_layer_diagnostic.py --mode select "$@"
    echo 'STAGE: Split selected recovery cases half/half; each worker tests all32 layers ×answer/audio ×-1.'
    run_pair steer "$@"
fi
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$cpu" -u phi4_all_layer_diagnostic.py --mode summary "$@"
