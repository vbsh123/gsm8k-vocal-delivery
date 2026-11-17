#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
candidate="${1:?Usage: bash setup_bbh_candidate.sh phi4|minicpm|flamingo}"
case "$candidate" in phi4|minicpm|flamingo) ;; *) echo 'Unknown model'; exit 2 ;; esac
command -v conda >/dev/null || { echo 'Conda is required for isolated model environments.'; exit 1; }
unset PYTHONHOME PYTHONPATH
export PIP_NO_CACHE_DIR=1
prefix="$PWD/.venv-bbh-$candidate"
if [[ "$candidate" == flamingo ]]; then
    pyver=3.11; torchver=2.9.1; visionver=0.24.1; cudaindex=cu128; transver=5.0.0
else
    pyver=3.10; torchver=2.6.0; visionver=0.21.0; cudaindex=cu124
    if [[ "$candidate" == phi4 ]]; then transver=4.48.2; else transver=4.51.0; fi
fi
if [[ ! -x "$prefix/bin/python" ]]; then
    conda create -y --prefix "$prefix" "python=$pyver" pip
fi
candidate_python="$prefix/bin/python"
if [[ "$candidate" == minicpm ]]; then
    pillowver=10.4.0; librosaver=0.9.0
else
    pillowver=11.1.0; librosaver=0.10.2.post1
fi
"$candidate_python" -c "import sys; assert sys.version_info[:2] == tuple(map(int, '$pyver'.split('.'))), sys.version"
"$candidate_python" -m pip install 'pip==25.3' 'setuptools==80.9.0' wheel
"$candidate_python" -m pip install "torch==$torchver" "torchaudio==$torchver" "torchvision==$visionver" \
    --index-url "https://download.pytorch.org/whl/$cudaindex"
"$candidate_python" -m pip install "transformers==$transver" 'accelerate==1.4.0' \
    'numpy==1.26.4' 'soundfile==0.13.1' 'scipy==1.15.2' "librosa==$librosaver" \
    "pillow==$pillowver" 'sentencepiece==0.2.0' 'tiktoken==0.9.0' 'einops==0.8.1'
if [[ "$candidate" == phi4 ]]; then
    "$candidate_python" -m pip install 'peft==0.13.2' 'backoff==2.2.1'
elif [[ "$candidate" == minicpm ]]; then
    "$candidate_python" -m pip install 'minicpmo-utils==1.0.6'
fi
"$candidate_python" - "$candidate" <<'PY'
import subprocess
import sys

checked = subprocess.run([sys.executable, '-m', 'pip', 'check'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
lines = [line.strip() for line in checked.stdout.splitlines() if line.strip()]
# Decord's old wheel tag can be rejected by pip despite its native library
# loading on this host. Accept only this exact diagnostic, after testing that
# library; preserve every dependency error and all other platform failures.
exception = 'decord 0.6.0 is not supported on this platform'
if checked.returncode and sys.argv[1] == 'minicpm' and lines == [exception]:
    import decord
    assert decord.__version__ == '0.6.0', decord.__version__
    assert callable(decord.VideoReader)
    print('Decord 0.6.0 native import passed; accepting its legacy wheel-tag diagnostic.')
else:
    print(checked.stdout, end='')
    if checked.returncode:
        raise SystemExit(checked.returncode)
PY
"$candidate_python" - "$candidate" <<'PY'
import sys, torch, transformers
from bbh_candidate_adapters import MODELS
spec = MODELS[sys.argv[1]]
assert transformers.__version__ == spec['transformers']
assert torch.__version__.split('+')[0] == spec['torch']
if sys.argv[1] == 'flamingo':
    from transformers import AudioFlamingo3ForConditionalGeneration
elif sys.argv[1] == 'minicpm':
    from transformers import Qwen3ForCausalLM
print('Environment ready:', sys.executable, 'torch', torch.__version__, 'transformers', transformers.__version__)
print('No model checkpoint downloaded by environment setup.')
PY
"$candidate_python" -m pip freeze > "$prefix/experiment-requirements.txt"
