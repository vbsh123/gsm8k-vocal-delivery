"""Read-only estimate before installing Audio Flamingo's environment/checkpoint."""
import os
from pathlib import Path
import shutil

from bbh_candidate_adapters import MODELS


def check():
    spec = MODELS['flamingo']
    root = Path(os.environ.get('HF_HUB_CACHE', '/workspace/.hf_home/hub'))
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root/('models--'+spec['model'].replace('/', '--'))/'snapshots'/spec['revision']
    present = sum(p.stat().st_size for p in snapshot.rglob('*.safetensors') if p.is_file())
    env = Path('.venv-bbh-flamingo')
    installed = sum(p.stat().st_size for p in env.rglob('*') if not p.is_symlink() and p.is_file()) if env.exists() else 0
    remaining_env = 0 if (env/'experiment-requirements.txt').exists() else max(0, 8*2**30-installed)
    missing_weights = max(0, spec['weights_gib']*2**30-present)
    required = missing_weights+remaining_env+4*2**30
    free = shutil.disk_usage(root).free
    print(f'Flamingo disk: {free/2**30:.1f} GiB free; estimate{required/2**30:.1f} GiB needed '
          f'(weights{missing_weights/2**30:.1f}, remaining environment{remaining_env/2**30:.1f}, reserve4).', flush=True)
    if free < required:
        raise RuntimeError('Not enough estimated space for Flamingo. No installation/download started and no data removed.')


if __name__ == '__main__': check()
