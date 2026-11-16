#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/kimi-bbh-layer-queue.lock
flock -n 9 || { echo 'The Kimi layer diagnostic queue is already running.'; exit 1; }
DIFFA_PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
if [[ -z "${KIMI_PYTHON:-}" ]]; then
    kimi_prefix="$(conda env list --json | "$DIFFA_PYTHON" -c 'import json,sys; print(next(p for p in json.load(sys.stdin)["envs"] if p.rstrip("/").endswith("/kimi-audio")))')"
    KIMI_PYTHON="$kimi_prefix/bin/python"
fi
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export PIP_NO_CACHE_DIR=1
if [[ -z "${HF_HOME:-}" && -d /workspace/.hf_home/hub ]]; then
    export HF_HOME=/workspace/.hf_home
fi
if [[ -n "${HF_HOME:-}" ]]; then
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
fi
for kind in ifeval ifbench mean-steering reasoning-transfer bbh-audio-models audio-models; do
    "$DIFFA_PYTHON" -u wait_for_experiments.py --kind "$kind"
done
echo 'Kimi BBH-3 layer diagnostic: native collection, base comparison, all 28 text layers x1.'
echo 'The base checkpoint needs about 17 GB extra disk. No cache or experiment files are deleted.'
"$KIMI_PYTHON" -u kimi_bbh_layer_diagnostic.py "$@"
"$DIFFA_PYTHON" -u kimi_bbh_layer_diagnostic.py "$@" --mode summary
