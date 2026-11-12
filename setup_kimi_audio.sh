#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PIP_NO_CACHE_DIR=1
command -v conda >/dev/null || { echo 'Conda is required for the separate kimi-audio environment.'; exit 1; }
SOURCE_DIR="$PWD/external/Kimi-Audio"
REVISION=349251e1d8f4f98d58fda59246381faecd7392e0
if [[ ! -d "$SOURCE_DIR" ]]; then
    git clone https://github.com/MoonshotAI/Kimi-Audio.git "$SOURCE_DIR"
fi
git -C "$SOURCE_DIR" checkout "$REVISION"
git -C "$SOURCE_DIR" submodule update --init --recursive
kimi_prefix="$(conda env list --json | python -c 'import json,sys; print(next((p for p in json.load(sys.stdin)["envs"] if p.rstrip("/").endswith("/kimi-audio")), ""))')"
if [[ -z "$kimi_prefix" ]]; then
    conda create -y -n kimi-audio python=3.10 pip
    kimi_prefix="$(conda env list --json | python -c 'import json,sys; print(next(p for p in json.load(sys.stdin)["envs"] if p.rstrip("/").endswith("/kimi-audio")))')"
fi
kimi_python="$kimi_prefix/bin/python"
unset PYTHONHOME PYTHONPATH
"$kimi_python" -c 'import sys; print("Kimi Python:", sys.executable, sys.version); assert sys.version_info[:2] == (3, 10), "Kimi requires the prepared Python 3.10 environment"'
"$kimi_python" -m pip install --upgrade pip 'setuptools<81' wheel
"$kimi_python" -m pip install \
    torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
# Official prebuilt wheel, selected for this Torch ABI; avoids local CUDA builds.
FLASH_WHEEL="$("$kimi_python" -c '
import platform, torch
assert platform.machine() == "x86_64", "The prepared wheel requires Linux x86_64"
assert torch.__version__.startswith("2.6.") and torch.version.cuda.startswith("12.")
abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
print("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abi"+abi+"-cp310-cp310-linux_x86_64.whl")
')"
"$kimi_python" -m pip install -r requirements-kimi.txt
"$kimi_python" -m pip install --no-deps "$FLASH_WHEEL"
"$kimi_python" -m pip install --no-deps --no-build-isolation -e "$SOURCE_DIR"
"$kimi_python" -u evaluate_kimi_audio.py --download-only
echo 'Ready for smoke: conda run --no-capture-output -n kimi-audio python -u evaluate_kimi_audio.py'
