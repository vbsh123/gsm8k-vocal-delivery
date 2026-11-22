"""Snapshot agreed DIFFA vectors, resumable aggregates, configs and saved answers; no audio/model weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from diffa_model_receipts import cached_model_revisions

ROOTS = (
    'diffa-gsm-shared-three-zeroshot-hashes', 'diffa-gsm-shared-three',
    'diffa-bbh-angry-only125', 'diffa-angry-only-remaining',
    'diffa-gsm-happy-agreed', 'diffa-angry-agreed/audit',
    'diffa-happy-winners', 'shared-three-emotions',
)
TEXT = {'.json', '.jsonl', '.csv', '.tsv', '.txt'}
STATES = {'style-vectors.pt', 'collection-stream.pt'}
SKIP = {'pairs', 'collection-cache', '__pycache__', '.git', 'classifier'}


def snapshot(path):
    data = path.read_bytes()
    if path.suffix == '.jsonl' and data and not data.endswith(b'\n'):
        # An actively appended last row is copied only after its newline is committed.
        data = data[:data.rfind(b'\n')+1]
    return data


def files_for(repo):
    pending = set()
    missing_roots = []

    def folder(path):
        if not path.is_dir():
            return
        for source in path.rglob('*'):
            if any(part in SKIP for part in source.relative_to(path).parts):
                continue
            if source.is_file() and (source.suffix in TEXT or source.name in STATES):
                pending.add(source.resolve())

    for name in ROOTS:
        path = repo/'outputs'/name
        if not path.exists():
            missing_roots.append(name)
        folder(path)
    # Original targets and vectors remain dependencies of the new frozen runs.
    catalog = repo/'winning_diffa_catalog.json'
    if catalog.is_file():
        for entry in json.loads(catalog.read_text()).values():
            for key in ('run', 'source'):
                if entry.get(key):
                    folder(repo/entry[key])
            for path in entry.get('reuse_runs', []):
                folder(repo/path)
    found = {}
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if not path.is_file():
            continue
        try:
            path.relative_to(repo/'outputs')
        except ValueError:
            continue
        data = snapshot(path)
        found[path] = data
        if path.name not in ('run_config.json', 'split.json', 'cohorts.json', 'data_config.json'):
            continue

        def dependencies(value):
            if isinstance(value, dict):
                for child in value.values():
                    dependencies(child)
            elif isinstance(value, list):
                for child in value:
                    dependencies(child)
            elif isinstance(value, str) and (value.startswith(str(repo/'outputs')+'/') or value.startswith('outputs/')):
                dep = Path(value) if Path(value).is_absolute() else repo/value
                if dep.is_file() and (dep.suffix in TEXT or dep.name in STATES):
                    pending.add(dep.resolve())
                    if dep.name in ('run_config.json', 'split.json'):
                        for sibling in ('results.jsonl', 'questions.jsonl', 'style-vectors.pt'):
                            if (dep.parent/sibling).is_file():
                                pending.add((dep.parent/sibling).resolve())
                elif dep.is_dir():
                    for name in ('run_config.json', 'split.json', 'results.jsonl', 'questions.jsonl', 'style-vectors.pt'):
                        if (dep/name).is_file():
                            pending.add((dep/name).resolve())
        dependencies(json.loads(data))
    # Keep the latest exact-hash emotional labels with the full inventory too.
    classifier = repo/'outputs/shared-three-emotions/classifier/results-emotion2vec.jsonl'
    if classifier.is_file():
        found[classifier] = snapshot(classifier)
    return found, missing_roots


def environment():
    command = [sys.executable, '-c',
               'import importlib.metadata as m,json,sys; print(json.dumps(dict(python=sys.version,packages={d.metadata["Name"]:d.version for d in m.distributions() if d.metadata.get("Name")})))']
    result = json.loads(subprocess.check_output(command, text=True))
    hub = Path(os.environ.get('HF_HUB_CACHE', str(Path(os.environ.get('HF_HOME', '/workspace/.hf_home'))/'hub')))
    revisions = cached_model_revisions(hub)
    result.update(hf_model_revisions=revisions, hf_cache=str(hub))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path('/workspace'))
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    files, missing = files_for(repo)
    if not files:
        raise ValueError('No existing runs found; run on the ORIGINAL DIFFA machine in the repository.')
    args.destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    final = args.destination/f'diffa-agreed-runs-{stamp}.zip'
    partial = final.with_suffix('.zip.partial')
    manifest = dict(repo=str(repo), git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_environment=environment(), missing_optional_roots=missing, files=[],
                    note='Point-in-time snapshot; unfinished collection-stream.pt is resumable, not a completed vector. '
                         'Original and agreed protocols stay in separate directories. No audio, weights, secrets or full per-pair activation caches.')
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path, data in sorted(files.items()):
            member = path.relative_to(repo).as_posix()
            archive.writestr(member, data)
            manifest['files'].append(dict(path=member, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        archive.writestr('DIFFA_MIGRATION_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(partial) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'ZIP verification failed: {bad}')
    partial.rename(final)
    digest = hashlib.sha256(final.read_bytes()).hexdigest()
    final.with_suffix('.zip.sha256').write_text(f'{digest}  {final.name}\n')
    print('Captured vector/collection status:')
    for path in sorted(p for p in files if p.name in STATES):
        print('  ', path.relative_to(repo))
    print(f'DOWNLOAD: {final}\nSIZE: {final.stat().st_size/2**20:.1f} MiB\nCHECKSUM: {final}.sha256')


if __name__ == '__main__':
    main()
