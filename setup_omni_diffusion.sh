#!/usr/bin/env bash
# CPU installation/import/download preflight; no language model is put on GPU.
set -euo pipefail
cd "$(dirname "$0")"
export PIP_NO_CACHE_DIR=1
export DS_BUILD_OPS=0
command -v conda >/dev/null || { echo 'Conda is required for the separate omni-diffusion environment.'; exit 1; }
if ! conda run -n omni-diffusion python -c 'import sys; assert sys.version_info[:2] == (3, 10)' >/dev/null 2>&1; then
    conda create -y -n omni-diffusion python=3.10 pip
fi
conda run --no-capture-output -n omni-diffusion python -m pip install --upgrade pip 'setuptools<81' wheel
conda run --no-capture-output -n omni-diffusion python -m pip install \
    torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu126
conda run --no-capture-output -n omni-diffusion python -m pip install \
    transformers==4.51.3 huggingface-hub==0.30.2 accelerate==1.8.1 funasr==1.2.6 \
    numpy==1.26.4 librosa==0.11.0 soundfile==0.13.1 scipy==1.15.3 \
    einops omegaconf sentencepiece kaldiio 'pillow>=10,<12'
conda run --no-capture-output -n omni-diffusion python -m pip install --no-build-isolation deepspeed==0.17.2
conda run --no-capture-output -n omni-diffusion python -u evaluate_omni_diffusion.py --download-only
echo 'Omni-Diffusion installation, imports, tokenizer, and checkpoint downloads completed. GPU inference is still to be tested.'
