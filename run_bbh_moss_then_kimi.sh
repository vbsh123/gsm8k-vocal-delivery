#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DIFFA_PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
mkdir -p outputs
exec 9>outputs/bbh-moss-kimi-queue.lock
flock -n 9 || { echo 'A MOSS/Kimi BBH queue is already running.'; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export PIP_NO_CACHE_DIR=1
if [[ -z "${HF_HOME:-}" ]]; then
    if [[ -d /workspace/.hf_home/hub ]]; then
        export HF_HOME=/workspace/.hf_home
    elif [[ -d /workspace/hf-cache/hub ]]; then
        export HF_HOME=/workspace/hf-cache
    else
        export HF_HOME="$PWD/outputs/hf-audio-models"
    fi
fi
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
if [[ "${1:-}" == '--regenerate-audio' ]]; then
    shift
    set -- --source outputs/azure-bbh-three-models-regenerated --output outputs/bbh-three-regenerated-models "$@"
    echo 'STAGE: Generate neutral/angry BBH audio locally; resumable Azure TTS, no GPU'
    "$DIFFA_PYTHON" -u evaluate_bbh_audio_models.py --model moss --mode generate "$@"
fi
echo 'STAGE: Check existing BBH audio before environment/model downloads'
"$DIFFA_PYTHON" -u evaluate_bbh_audio_models.py --model moss --mode prepare "$@"
echo 'STAGE: Wait for current experiment queues on this machine'
for kind in reasoning-transfer mean-steering ifeval ifbench audio-models pitch-speed classification; do
    "$DIFFA_PYTHON" -u wait_for_experiments.py --kind "$kind"
done
"$DIFFA_PYTHON" -c 'from lm_eval.api.task import ConfigurableTask'
echo 'STAGE: Set up MOSS in its own environment'
bash setup_moss_audio.sh
echo 'STAGE: MOSS-Audio-8B-Instruct neutral/angry BBH evaluation; no steering'
LD_LIBRARY_PATH="$PWD/.venv-moss/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    .venv-moss/bin/python -u evaluate_bbh_audio_models.py --model moss "$@"
"$DIFFA_PYTHON" -u evaluate_bbh_audio_models.py --model moss --mode summary "$@"
echo 'STAGE: Find or prepare the existing Kimi environment'
kimi_prefix="$(conda env list --json | "$DIFFA_PYTHON" -c 'import json,sys; print(next((p for p in json.load(sys.stdin)["envs"] if p.rstrip("/").endswith("/kimi-audio")), ""))')"
if [[ -z "$kimi_prefix" || ! -d external/Kimi-Audio/.git ]] || ! "$kimi_prefix/bin/python" -c 'import flash_attn, transformers; from kimia_infer.api.kimia import KimiAudio; assert transformers.__version__ == "4.51.3"' >/dev/null 2>&1; then
    bash setup_kimi_audio.sh
    kimi_prefix="$(conda env list --json | "$DIFFA_PYTHON" -c 'import json,sys; print(next(p for p in json.load(sys.stdin)["envs"] if p.rstrip("/").endswith("/kimi-audio")))')"
fi
echo 'STAGE: Kimi-Audio neutral/angry BBH evaluation; no steering'
"$kimi_prefix/bin/python" -u evaluate_bbh_audio_models.py --model kimi "$@"
"$DIFFA_PYTHON" -u evaluate_bbh_audio_models.py --model kimi --mode summary "$@"
echo 'Finished: MOSS then Kimi. Original audio and earlier results retained.'
