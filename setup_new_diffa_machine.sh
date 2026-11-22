#!/usr/bin/env bash
# Run inside /workspace/gsm8k-vocal-delivery on the NEW A40 only.
set -euo pipefail
cd "$(dirname "$0")"
if [[ "$PWD" != /workspace/gsm8k-vocal-delivery ]]; then
  echo 'Clone into /workspace/gsm8k-vocal-delivery to preserve frozen absolute paths.' >&2
  exit 1
fi
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME=/workspace/.hf_home
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PIP_NO_CACHE_DIR=1
mkdir -p outputs "$HF_HOME"
exec 9>outputs/new-diffa-machine-setup.lock
flock -n 9 || { echo 'Machine setup is already running.'; exit 1; }
if [[ $# -gt 0 ]]; then
  archives=("$@")
else
  shopt -s nullglob
  audio=(/workspace/jenny-six-datasets-audio-*.zip)
  vectors=(/workspace/diffa-agreed-runs-*.zip)
  if [[ ${#audio[@]} -ne 1 || ${#vectors[@]} -gt 1 ]]; then
    echo 'Upload the audio ZIP completely, then pass explicit ZIP paths if more than one exists.' >&2
    exit 1
  fi
  archives=("${audio[@]}" "${vectors[@]}")
fi
python="${PYTHON_BIN:-python3}"
"$python" restore_diffa_bundle.py "${archives[@]}"
echo 'STAGE: Install pinned DIFFA source and CUDA environment.'
bash setup_diffa.sh
echo 'STAGE: Install and verify the pinned scoring harness.'
bash setup_harness.sh
echo 'STAGE: Restore recorded auxiliary IFEval package versions when the run bundle is present.'
.venv-diffa/bin/python - <<'PY'
import json
from pathlib import Path
import subprocess
import sys
manifest = Path('DIFFA_MIGRATION_MANIFEST.json')
if manifest.exists():
    packages = {k.lower().replace('_', '-'): v for k,v in json.loads(manifest.read_text())['source_environment']['packages'].items()}
    requirements = [f'{name}=={packages[name]}' for name in ('absl-py', 'langdetect', 'immutabledict') if name in packages]
    if requirements:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', *requirements], check=True)
else:
    print('Upload/restore the run bundle, then rerun setup to compare the original environment before experiments.')
PY
echo 'STAGE: Check restored runs and software receipt; no model inference.'
.venv-diffa/bin/python -u check_diffa_migration.py
echo 'Setup complete. Model weights download on first load. No experiment has been started.'
echo 'If the vector bundle is uploaded later: python3 restore_diffa_bundle.py /workspace/diffa-agreed-runs-*.zip'
