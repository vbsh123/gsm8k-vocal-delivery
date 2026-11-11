#!/usr/bin/env bash
# Remaining speed validation, then existing-dataset emotion audits. No Qwen/Kimi.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 6>outputs/speeds-and-classification.lock
flock 6
export PIP_NO_CACHE_DIR=1

status=outputs/speeds-and-classification-status.tsv
run_stage() {
  local name="$1"
  shift
  echo "$(date -Is) START $name"
  if "$@"; then
    printf '%s\t%s\tcomplete\n' "$(date -Is)" "$name" | tee -a "$status"
  else
    printf '%s\t%s\tFAILED\n' "$(date -Is)" "$name" | tee -a "$status"
    return 1
  fi
}
retry() {
  for attempt in 1 2 3; do
    if "$@"; then return 0; fi
    echo "Attempt $attempt failed; saved progress is retained."
    sleep 5
  done
  return 1
}

if [[ ! -x .venv-diffa/bin/python || ! -f outputs/spoken-gsm-audiostretchy200/run_config.json ]]; then
  echo 'Run this on the machine with the completed 200-question speed sweep and .venv-diffa.' >&2
  exit 1
fi
.venv-diffa/bin/python -m pip install audiostretchy==1.3.14
run_stage remaining-speeds bash run_audiostretchy_speed_remaining.sh || true

if ! .venv-emotion/bin/python -c 'import torch, torchaudio, funasr, librosa, soundfile' >/dev/null 2>&1; then
  if ! run_stage setup-emotion retry env PYTHON_BIN="$PWD/.venv-diffa/bin/python" bash setup_emotion.sh; then
    echo 'Emotion environment setup failed; inspect the log. Speed results were retained.' >&2
    exit 1
  fi
fi
run_stage setup-audit-dependencies retry .venv-emotion/bin/python -m pip install pyarrow==19.0.1 'fsspec[http]' gdown
.venv-emotion/bin/python -c 'import numpy, pyarrow.parquet; print("Parquet imports OK; NumPy", numpy.__version__)'
export STREAM_AUDIO=1
for dataset in spoken-gsm mmau mmau-pro; do
  run_stage "classify-$dataset" retry bash run_dataset_emotions.sh cuda:0 "$dataset" || true
done
if run_stage prepare-speechr retry .venv-emotion/bin/python -u prepare_speechr_audio.py; then
  run_stage classify-speechr retry bash run_dataset_emotions.sh cuda:0 speechr || true
fi
echo "Finished queue. Per-stage status: $status"
cat "$status"
