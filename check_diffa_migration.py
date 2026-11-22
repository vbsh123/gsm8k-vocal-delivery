"""Read-only overview of restored agreed DIFFA vectors and software versions."""
import importlib.metadata
import json
from pathlib import Path

from pack_diffa_migration import ROOTS


def main():
    import torch
    print('CUDA:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
    manifest = Path('DIFFA_MIGRATION_MANIFEST.json')
    if manifest.exists():
        previous = json.loads(manifest.read_text())['source_environment']
        packages = {name.lower(): value for name, value in previous['packages'].items()}
        print('Source Python:', previous['python'].splitlines()[0])
        print('Pinned source-model revisions:', previous.get('hf_model_revisions', {}))
        for name in ('torch', 'torchaudio', 'transformers', 'numpy', 'peft', 'accelerate', 'lm_eval', 'soundfile'):
            try:
                current = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                current = 'MISSING'
            original = packages.get(name.lower(), packages.get(name.replace('_', '-').lower(), 'not recorded'))
            print(f'{name:16} original={original:20} new={current}')
    else:
        print('Vector bundle has not been restored yet; audio alone does not contain experiment vectors.')
    print('\nRestored new-vector runs (completed vector versus resumable collection):')
    for name in ROOTS[:5]:
        root = Path('outputs')/name
        if not root.exists():
            continue
        configs = sorted(root.rglob('run_config.json'))
        for config in configs:
            if not (config.parent/'style-vectors.pt').is_file() and not (config.parent/'collection-stream.pt').is_file():
                continue
            plan = json.loads(config.read_text())
            status = 'VECTOR COMPLETE' if (config.parent/'style-vectors.pt').is_file() else 'COLLECTION CHECKPOINT'
            print(f'{config.parent}: {status}; collection={len(plan.get("collection_ids", []))}; test={len(plan.get("test_ids", []))}; emotion={plan.get("emotion")}')
    print('Run the saved baseline replay before treating a cross-machine comparison as controlled.')


if __name__ == '__main__':
    main()
