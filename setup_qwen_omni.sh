#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PIP_NO_CACHE_DIR=1
command -v conda >/dev/null || { echo 'Conda is required to create the separate qwen-omni environment.'; exit 1; }
qwen_prefix="$(conda env list --json | python -c 'import json,sys; print(next((p for p in json.load(sys.stdin)["envs"] if p.rstrip("/").endswith("/qwen-omni")), ""))')"
if [[ -z "$qwen_prefix" ]]; then
    conda create -y -n qwen-omni python=3.10 pip
    qwen_prefix="$(conda env list --json | python -c 'import json,sys; print(next(p for p in json.load(sys.stdin)["envs"] if p.rstrip("/").endswith("/qwen-omni")))')"
fi
qwen_python="$qwen_prefix/bin/python"
unset PYTHONHOME PYTHONPATH
"$qwen_python" -c 'import sys; print("Qwen Python:", sys.executable, sys.version); assert sys.version_info[:2] == (3, 10), "Qwen requires the prepared Python 3.10 environment"'
"$qwen_python" -m pip install --upgrade pip
"$qwen_python" -m pip install \
    torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
"$qwen_python" -m pip install \
    transformers==4.52.4 accelerate==1.8.1 numpy==1.26.4 \
    librosa==0.11.0 soundfile==0.13.1 scipy==1.15.3 'pillow>=10,<12'
"$qwen_python" -c \
    'import torch; from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor; print("Qwen imports OK; torch", torch.__version__, "CUDA available:", torch.cuda.is_available())'
echo 'Ready: conda run --no-capture-output -n qwen-omni python -u evaluate_qwen_omni.py'
