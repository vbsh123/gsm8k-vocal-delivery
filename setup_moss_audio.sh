#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PIP_NO_CACHE_DIR=1
unset PYTHONHOME PYTHONPATH
command -v conda >/dev/null || { echo 'Conda is required for the separate MOSS environment.'; exit 1; }
source_dir="$PWD/external/MOSS-Audio"
revision=66326e6e0db34f036c86a76ba005efa4830c69dd
mkdir -p external
if [[ ! -d "$source_dir/.git" ]]; then
    git clone https://github.com/OpenMOSS/MOSS-Audio.git "$source_dir"
fi
git -C "$source_dir" checkout "$revision"
if [[ ! -x "$PWD/.venv-moss/bin/python" ]]; then
    conda create -y --prefix "$PWD/.venv-moss" python=3.12 pip
fi
conda install -y --prefix "$PWD/.venv-moss" -c conda-forge 'ffmpeg=7'
export LD_LIBRARY_PATH="$PWD/.venv-moss/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
moss_python="$PWD/.venv-moss/bin/python"
"$moss_python" -c 'import sys; print("MOSS Python:", sys.executable, sys.version); assert sys.version_info[:2] == (3, 12)'
"$moss_python" -m pip install --upgrade pip
"$moss_python" -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    -e "$source_dir[torch-runtime]"
"$moss_python" - <<'PY'
import tempfile
from pathlib import Path
import wave
import torch, transformers
from moss_audio_io import load_audio
from src.modeling_moss_audio import MossAudioModel
from src.processing_moss_audio import MossAudioProcessor
assert transformers.__version__ == '4.57.1'
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)/'long.wav'
    with wave.open(str(path), 'wb') as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(24000)
        audio.writeframes(b'\0\0'*(40*24000))
    waveform = load_audio(str(path), sample_rate=16000)
    assert len(waveform) == 40*16000, 'Full-audio decode/resampling preflight failed'
print('SoundFile WAV decode + MOSS mono/TorchAudio resampling verified: full 40 seconds preserved.')
print('MOSS imports OK:', torch.__version__, transformers.__version__, 'CUDA available:', torch.cuda.is_available())
PY
echo 'Ready: .venv-moss/bin/python evaluate_bbh_audio_models.py --model moss'
