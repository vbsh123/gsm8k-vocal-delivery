#!/usr/bin/env bash
# IndexTTS-2.5 has its own Python 3.11 environment.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
index_root="$PWD/external/index-tts"
index_commit="d9e41aac89fd00b3d71497fddb287b7f24613712"

apt-get update
apt-get install -y ffmpeg libsox-dev build-essential
if [[ ! -e "$index_root" ]]; then
  mkdir -p external
  git clone --filter=blob:none --no-checkout https://github.com/index-tts/index-tts.git "$index_root"
  git -C "$index_root" checkout --detach "$index_commit"
fi

# Bootstrap uv outside the target env so it can provision Python 3.11 itself.
"$PYTHON_BIN" -m venv "$index_root/.bootstrap"
"$index_root/.bootstrap/bin/python" -m pip install --upgrade uv
UV_PROJECT_ENVIRONMENT="$PWD/.venv-index" "$index_root/.bootstrap/bin/uv" sync \
  --project "$index_root" --python 3.11 --no-dev

.venv-index/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    "IndexTeam/IndexTTS-2.5", local_dir=Path("external/index-tts/checkpoints"),
    ignore_patterns=["qwen0.6bemo4-merge/*"],
)
PY
printf '\nReady: .venv-index/bin/python index_smoke.py\n'
