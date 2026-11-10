#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs

if [[ ! -f outputs/spoken-gsm-original/manifest.jsonl ]]; then
  .venv-diffa/bin/python -u prepare_spoken_gsm.py --num-examples 300
fi

.venv-diffa/bin/python -u speed_listen5.py --output outputs/speed-listen5
echo "Listening preview ready: outputs/speed-listen5.zip"

exec 9>outputs/jenny-task-vectors.lock
echo "Waiting for the GPU queue lock."
flock 9
while true; do
  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
  if [[ -z "$gpu_pids" ]]; then break; fi
  echo "Waiting for GPU PID(s): $gpu_pids"
  sleep 30
done

.venv-diffa/bin/python -u evaluate_audio_speed.py \
  --num-examples 100 \
  --rates 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 \
  --reuse-from outputs/spoken-gsm-speed100 \
  --output outputs/spoken-gsm-speed100-extended --save-audio

flock -u 9
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv-diffa/bin/python -u audit_speed_pitch.py \
    --speed-run outputs/spoken-gsm-speed100-extended \
    --output outputs/spoken-gsm-speed100-extended-pitch-audit \
    --num-examples 5 --listen-examples 5

.venv-diffa/bin/python - <<'PACK'
from pathlib import Path
import zipfile
folder=Path('outputs/spoken-gsm-speed100-extended-pitch-audit')
archive=folder.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in folder.rglob('*'):
        if path.is_file(): z.write(path,path.relative_to(folder.parent))
print('Download:',archive)
PACK
