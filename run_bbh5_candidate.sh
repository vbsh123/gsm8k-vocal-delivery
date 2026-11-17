#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
candidate="${1:-phi4}"
if [[ $# -gt 0 ]]; then shift; fi
case "$candidate" in phi4|minicpm|flamingo) ;; *) echo 'Unknown candidate'; exit 2 ;; esac
mkdir -p outputs
exec 9>"outputs/bbh5-candidate-$candidate.lock"
flock -n 9 || { echo 'This BBH-five queue is already running.'; exit 1; }
DIFFA_PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
"$DIFFA_PYTHON" -u evaluate_bbh5_candidates.py --model "$candidate" --mode audio "$@"
"$DIFFA_PYTHON" -u evaluate_bbh5_candidates.py --model "$candidate" --mode prepare "$@"
"$DIFFA_PYTHON" -c 'from lm_eval.api.task import ConfigurableTask'
echo 'STAGE: Wait for GSM8K, BBH-three and Kimi queues'
flock "outputs/gsm-candidate-$candidate.lock" true
flock "outputs/bbh-candidate-$candidate.lock" true
for file in kimi-bbh5-queue kimi-multiarith-queue kimi-gsm8k-queue; do flock "outputs/$file.lock" true; done
if [[ ! -x "$PWD/.venv-bbh-$candidate/bin/python" ]]; then bash setup_bbh_candidate.sh "$candidate"; fi
echo 'STAGE: BBH-five native audio evaluation, both prompts, no steering'
"$PWD/.venv-bbh-$candidate/bin/python" -u evaluate_bbh5_candidates.py --model "$candidate" --mode evaluate "$@"
"$DIFFA_PYTHON" -u evaluate_bbh5_candidates.py --model "$candidate" --mode summary "$@"
