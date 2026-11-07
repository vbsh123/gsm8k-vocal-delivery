#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PIP_NO_CACHE_DIR=1
command -v conda >/dev/null || { echo 'Conda is required to create the separate qwen-omni environment.'; exit 1; }
if ! conda run -n qwen-omni python -c 'import sys; assert sys.version_info[:2] == (3, 10)' >/dev/null 2>&1; then
    conda create -y -n qwen-omni python=3.10 pip
fi
conda run --no-capture-output -n qwen-omni python -m pip install --upgrade pip
conda run --no-capture-output -n qwen-omni python -m pip install \
    torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
conda run --no-capture-output -n qwen-omni python -m pip install \
    transformers==4.52.4 accelerate==1.8.1 numpy==1.26.4 \
    librosa==0.11.0 soundfile==0.13.1 scipy==1.15.3 'pillow>=10,<12'
conda run --no-capture-output -n qwen-omni python -c \
    'import torch; from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor; print("Qwen imports OK; torch", torch.__version__, "CUDA available:", torch.cuda.is_available())'
echo 'Ready: conda run --no-capture-output -n qwen-omni python -u evaluate_qwen_omni.py'
