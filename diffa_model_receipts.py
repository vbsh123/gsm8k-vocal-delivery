"""Read exact source-model revisions without loading models or importing GPU libraries."""
import json
from pathlib import Path
import re

MODELS = ('GSAI-ML/LLaDA-8B-Instruct', 'openai/whisper-large-v3', 'zhoujiaming777/DIFFA-2')


def cached_model_revisions(hub):
    revisions = {}
    for model in MODELS:
        cache = Path(hub)/('models--'+model.replace('/','--'))
        ref = cache/'refs/main'
        value = None
        if ref.is_file():
            value = ref.read_text().strip()
        elif (cache/'snapshots').is_dir():
            snapshots = [p.name for p in (cache/'snapshots').iterdir() if p.is_dir()]
            if len(snapshots) == 1:
                value = snapshots[0]
        if value:
            if not re.fullmatch('[0-9a-f]{40}', value):
                raise ValueError(f'Expected immutable model commit SHA, got {value!r}: {cache}')
            revisions[model] = value
    return revisions


def model_revisions():
    receipts = []
    local = Path('outputs/diffa-model-revisions.json')
    if local.exists():
        receipts.append(json.loads(local.read_text())['models'])
    migration = Path('DIFFA_MIGRATION_MANIFEST.json')
    if migration.exists():
        values = json.loads(migration.read_text()).get('source_environment', {}).get('hf_model_revisions', {})
        if values:
            receipts.append(values)
    if len(receipts)>1 and any(receipt != receipts[0] for receipt in receipts[1:]):
        raise ValueError('Source-model revision receipts conflict; inspect before loading weights.')
    return receipts[0] if receipts else {}
