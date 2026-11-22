#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/finish-jenny-audio.lock
flock -n 9 || { echo 'Jenny audio completion is already running.'; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
echo 'STAGE: Fill missing GSM neutral and SVAMP angry; reuse existing recordings.'
"$python" -u generate_missing_emotions.py --datasets gsm8k svamp --styles neutral angry --run
echo 'STAGE: Classify missing emotional clips on CPU; no wait for DIFFA GPU.'
bash run_shared_emotions.sh --device cpu
echo 'STAGE: Verify all Jenny audio hashes, coverage and WAV payloads.'
"$python" -u check_transfer_audio.py
"$python" -c 'import json; from pathlib import Path; s=json.loads(Path("outputs/audio-transfer-check/summary.json").read_text()); missing={n:{style:len(ids) for style,ids in r["missing"].items() if ids} for n,r in s["datasets"].items()}; missing={n:r for n,r in missing.items() if r}; assert not missing, f"Still missing audio; archive not started: {missing}"'
echo 'STAGE: ZIP validated Jenny audio with plans and labels; print download path and size.'
"$python" -u pack_jenny_audio.py
