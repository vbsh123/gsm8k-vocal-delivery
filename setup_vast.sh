#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ "${1:-}" == "--diffa" ]]; then
  exec bash setup_diffa.sh
elif [[ $# -gt 0 ]]; then
  printf 'Usage: bash setup_vast.sh [--diffa]\n' >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info < (3, 14), "Use Python 3.10–3.13 (3.12 recommended)"'
nvidia-smi
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# CUDA 12.8 wheels need a compatible NVIDIA driver on the Vast host.
.venv/bin/python -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check host driver and PyTorch wheels"; print("Ready:", torch.cuda.get_device_name(0))'
printf '\nSetup complete. Run: source .venv/bin/activate\n'
