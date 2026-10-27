#!/usr/bin/env bash
# DIFFA-2 source and environment only; evaluation stays in evaluate.py.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "DIFFA needs Python 3.10–3.12; set PYTHON_BIN=python3.12 if necessary"'
nvidia-smi

# Use the same source revision as our caching experiments.
diffa_commit="0a87594e110984fbaf00ebf5bfe5840c212e4438"
diffa_root="$PWD/external/DIFFA"
if [[ ! -e "$diffa_root" ]]; then
  mkdir -p external
  git clone --no-checkout https://github.com/NKU-HLT/DIFFA.git "$diffa_root"
  git -C "$diffa_root" checkout --detach "$diffa_commit"
elif [[ "$(git -C "$diffa_root" rev-parse HEAD)" != "$diffa_commit" ]]; then
  printf 'Existing DIFFA source has a different revision; left unchanged: %s\n' "$diffa_root" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv .venv-diffa
.venv-diffa/bin/python -m pip install --upgrade pip
.venv-diffa/bin/python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv-diffa/bin/python -m pip install -r requirements-diffa.txt
.venv-diffa/bin/python -m pip check

# Register the official source in this venv, so imports work from evaluate.py
# without a manual PYTHONPATH export or modifications to upstream files.
.venv-diffa/bin/python - "$diffa_root/DIFFA-2" <<'PY'
from pathlib import Path
import sys
import sysconfig

source = Path(sys.argv[1]).resolve()
site_packages = Path(sysconfig.get_path("purelib"))
(site_packages / "diffa_source.pth").write_text(str(source) + "\n", encoding="utf-8")
print("Registered DIFFA-2 source:", source)
PY

PYTHONDONTWRITEBYTECODE=1 .venv-diffa/bin/python - <<'PY'
import torch
from src.configuration_DIFFA import DIFFAConfig
from src.modeling_DIFFA import DIFFAModel
from src.fast_dllm.model.modeling_llada import LLaDAModelLM

assert torch.cuda.is_available(), "CUDA unavailable: check host driver and PyTorch wheels"
print("DIFFA-2 imports OK. GPU:", torch.cuda.get_device_name(0))
PY
printf '\nDIFFA-2 setup complete. Run: source .venv-diffa/bin/activate\n'
printf 'Model weights will download when your evaluation code loads them.\n'
