#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
candidate="${1:?Usage: bash run_bbh_candidate.sh phi4|minicpm|flamingo [evaluation options]}"
shift
case "$candidate" in phi4|minicpm|flamingo) ;; *) echo 'Unknown model'; exit 2 ;; esac
mkdir -p outputs
exec 9>"outputs/bbh-candidate-$candidate.lock"
flock -n 9 || { echo "The $candidate queue is already running."; exit 1; }
DIFFA_PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
if [[ -z "${HF_HOME:-}" ]]; then export HF_HOME=/workspace/.hf_home; fi
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
echo 'STAGE: Reuse/resume BBH-three audio and freeze both evaluation protocols'
"$DIFFA_PYTHON" -u evaluate_bbh_candidates.py --model "$candidate" --mode audio "$@"
"$DIFFA_PYTHON" -u evaluate_bbh_candidates.py --model "$candidate" --mode prepare "$@"
"$DIFFA_PYTHON" -c 'from lm_eval.api.task import ConfigurableTask'
echo 'STAGE: Wait for current Kimi queues; no processes killed'
for file in kimi-bbh5-queue kimi-multiarith-queue kimi-gsm8k-queue kimi-bbh-full-steering-queue kimi-bbh-followups-queue kimi-bbh-explicit-reasoning-queue; do
    flock "outputs/$file.lock" true
done
mkdir -p outputs/bbh-three-explicit-reasoning/kimi
flock outputs/bbh-three-explicit-reasoning/kimi/queue.lock true
echo "STAGE: Prepare only the chosen $candidate environment"
bash setup_bbh_candidate.sh "$candidate"
echo "STAGE: $candidate native audio evaluation; GPU lock acquired by evaluator"
"$PWD/.venv-bbh-$candidate/bin/python" -u evaluate_bbh_candidates.py --model "$candidate" --mode evaluate "$@"
"$DIFFA_PYTHON" -u evaluate_bbh_candidates.py --model "$candidate" --mode summary "$@"
