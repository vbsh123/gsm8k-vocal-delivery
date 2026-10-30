#!/usr/bin/env bash
# Separate Fish environment; the Qwen and DIFFA environments are left alone.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12: PYTHON_BIN=python3.12 bash setup_fish.sh"'

apt-get update
apt-get install -y portaudio19-dev libsox-dev ffmpeg build-essential

fish_root="$PWD/external/fish-speech"
fish_commit="0f8a12e8a651062fdee9c33a894bdccd3a8513f1"
if [[ ! -e "$fish_root" ]]; then
  mkdir -p external
  git clone --filter=blob:none --no-checkout https://github.com/fishaudio/fish-speech.git "$fish_root"
  git -C "$fish_root" checkout --detach "$fish_commit"
fi

"$PYTHON_BIN" -m venv .venv-fish
.venv-fish/bin/python -m pip install --upgrade pip uv
UV_PROJECT_ENVIRONMENT="$PWD/.venv-fish" .venv-fish/bin/uv sync \
  --project "$fish_root" --python "$PYTHON_BIN" --extra cu128 --no-dev --inexact

.venv-fish/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    "fishaudio/s2-pro", local_dir=Path("external/fish-speech/checkpoints/s2-pro"),
    allow_patterns=["*.json", "*.safetensors", "codec.pth", "*.jinja"],
)
PY
printf '\nReady: .venv-fish/bin/python fish_smoke.py\n'
