#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/diffa-voice-transfer-wrapper.lock
flock -n 9 || { echo 'New voice/transfer queue is already running.'; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
echo 'STAGE: New Guy voice; same290 MultiArith test questions; generate/resume580 clips on CPU.'
.venv-diffa/bin/python -u run_multiarith_second_voice.py --mode prepare >outputs/multiarith-guy.log 2>&1
echo 'STAGE: New-voice DIFFA neutral/angry evaluation; waits for shared GPU.'
.venv-diffa/bin/python -u run_multiarith_second_voice.py --mode evaluate >>outputs/multiarith-guy.log 2>&1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u run_multiarith_second_voice.py --mode summary >>outputs/multiarith-guy.log 2>&1
echo 'STAGE: Every original task vector to every OTHER task at L31 answer x1.'
bash run_winning_transfer.sh >outputs/winning-transfer.log 2>&1
echo 'Finished new voice and transfer queue.'
