#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
device="${1:-cuda:0}"
if (( $# )); then shift; fi
datasets=("$@")
if (( ${#datasets[@]} == 0 )); then datasets=(spoken-gsm mmau mmau-pro speechr); fi
for dataset in "${datasets[@]}"; do
  if [[ "$dataset" == speechr && ! -d outputs/speechr ]]; then
    echo 'SpeechR folder missing: copy/extract the downloaded audio into outputs/speechr first.' >&2
    exit 1
  fi
done
if [[ "$device" != cpu ]]; then
  exec 9>outputs/jenny-task-vectors.lock
  echo "Waiting for the GPU queue lock."
  flock 9
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
    echo "Waiting for active GPU work."
    sleep 30
  done
fi
for dataset in "${datasets[@]}"; do
  extra=()
  if [[ "$dataset" == speechr ]]; then extra=(--audio-root outputs/speechr); fi
  for model in emotion2vec sensevoice; do
    echo "Dataset emotion audit: $dataset / $model / $device"
    .venv-emotion/bin/python -u classify_dataset_emotions.py \
      --dataset "$dataset" --model "$model" --device "$device" ${STREAM_AUDIO:+--stream-audio} "${extra[@]}"
  done
done
