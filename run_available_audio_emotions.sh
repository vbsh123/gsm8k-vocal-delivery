#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 8>outputs/available-audio-emotions-wrapper.lock
flock -n 8 || { echo 'All-audio emotion audit is already running.'; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
device="${1:-cuda:0}"
echo 'STAGE: Wait for active TTS preparation only; no transfer/GPU queue wait.'
.venv-diffa/bin/python -u wait_for_experiments.py --kind emotion-audio-preparation
echo 'STAGE: Inventory all surviving experiment audio; hash/deduplicate; record missing references.'
.venv-emotion/bin/python -u audit_available_audio.py --mode inventory
for model in emotion2vec sensevoice; do
    echo "STAGE: $model / $device; full recordings, all styles; no filtering."
    .venv-emotion/bin/python -u audit_available_audio.py --mode classify --model "$model" --device "$device"
done
echo 'STAGE: Summary and evidence ZIP including strengths, voice, transfer, both classifiers and samples.'
.venv-emotion/bin/python -u audit_available_audio.py --mode summary
.venv-diffa/bin/python -u pack_experiment_backup.py --only strengths voice transfer emotions --destination /workspace --samples 3 --sample-mb 100
echo 'Finished. Download the ZIP and SHA256 paths printed above; inspect classifier errors/missing audio in the audit manifest.'
