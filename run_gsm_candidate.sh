#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
candidate="${1:-phi4}"
if [[ $# -gt 0 ]]; then shift; fi
case "$candidate" in phi4|minicpm|flamingo) ;; *) echo 'Unknown candidate'; exit 2 ;; esac
mkdir -p outputs
exec 9>"outputs/gsm-candidate-$candidate.lock"
flock -n 9 || { echo 'This GSM8K candidate queue is already running.'; exit 1; }
DIFFA_PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
echo 'STAGE: Reuse all 1269 neutral/angry GSM8K audio pairs and freeze prompts'
"$DIFFA_PYTHON" -u evaluate_gsm_candidates.py --model "$candidate" --mode audio "$@"
"$DIFFA_PYTHON" -u evaluate_gsm_candidates.py --model "$candidate" --mode prepare "$@"
"$DIFFA_PYTHON" -c 'from lm_eval.api.task import ConfigurableTask'
echo 'STAGE: Wait for current candidate BBH and Kimi queues'
flock "outputs/bbh-candidate-$candidate.lock" true
for file in kimi-bbh5-queue kimi-multiarith-queue kimi-gsm8k-queue; do flock "outputs/$file.lock" true; done
if [[ ! -x "$PWD/.venv-bbh-$candidate/bin/python" ]]; then bash setup_bbh_candidate.sh "$candidate"; fi
echo 'STAGE: Native full-audio GSM8K evaluation; no steering'
"$PWD/.venv-bbh-$candidate/bin/python" -u evaluate_gsm_candidates.py --model "$candidate" --mode evaluate "$@"
"$DIFFA_PYTHON" -u evaluate_gsm_candidates.py --model "$candidate" --mode summary "$@"
