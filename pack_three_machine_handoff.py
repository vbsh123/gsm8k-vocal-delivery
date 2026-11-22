"""Wait for complete shared-GSM vectors and export a no-audio handoff for the new machine."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import zipfile

from pack_diffa_migration import environment, files_for, snapshot

GSM = Path('outputs/diffa-gsm-shared-three-zeroshot-hashes')
NEW_ROOTS = ('diffa-gsm-shared-three-zeroshot-hashes', 'diffa-bbh-angry-only125',
             'diffa-angry-only-remaining', 'diffa-shared-three-tasks', 'diffa-gsm-happy-agreed')


def select_handoff(files, repo):
    return {path: data for path, data in files.items() if path.suffix != '.pt' or
            path.name == 'style-vectors.pt' and any((repo/'outputs'/name) in path.parents for name in NEW_ROOTS)}


def worker_matches(argv, cwd, repo):
    if not any(Path(arg).name == 'diffa_gsm_shared_emotions.py' for arg in argv):
        return False
    def option(flag, default):
        for index, value in enumerate(argv):
            if value == flag and index+1 < len(argv):
                return argv[index+1]
            if value.startswith(flag+'='):
                return value.split('=', 1)[1]
        return default
    if option('--mode', 'run') != 'run':
        return False
    output = Path(option('--output', str(GSM)))
    return (output if output.is_absolute() else cwd/output).resolve() == (repo/GSM).resolve()


def complete():
    from winning_diffa_sweeps import read_vector
    for emotion in ('angry', 'happy'):
        root = GSM/emotion
        if not all((root/name).exists() for name in ('run_config.json', 'style-vectors.pt', 'collection-storage.json')):
            return False
        plan = json.loads((root/'run_config.json').read_text())
        try:
            storage = json.loads((root/'collection-storage.json').read_text())
        except json.JSONDecodeError:
            return False
        if storage['completed_pairs'] != len(plan['collection_ids']):
            return False
        _, receipt = read_vector(root/'style-vectors.pt', emotion)
        if (receipt['extraction_ids'] != plan['collection_ids'] or receipt['pairs'] != len(plan['collection_ids'])
                or receipt['collection_prompt'] != plan['collection_prompt']):
            raise ValueError(f'Invalid completed GSM vector: {emotion}')
    return True


def stop_only_shared_gsm_worker(repo):
    stopped = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            argv = [x.decode() for x in (process/'cmdline').read_bytes().split(b'\0') if x]
            cwd = (process/'cwd').resolve()
            if not worker_matches(argv, cwd, repo):
                continue
            os.kill(int(process.name), signal.SIGTERM)
            stopped.append(int(process.name))
        except (OSError, ValueError, IndexError):
            continue
    print('Shared-GSM worker handed off after completed collection; stopped PIDs:', stopped, flush=True)
    for _ in range(60):
        alive = []
        for pid in stopped:
            path = Path('/proc')/str(pid)/'stat'
            try:
                if path.read_text().split()[2] != 'Z':
                    alive.append(pid)
            except OSError:
                continue
        if not alive:
            return
        time.sleep(1)
    raise RuntimeError('Shared-GSM worker has not exited; no ZIP snapshot made yet.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true', help='Wait until both shared GSM collections are complete.')
    parser.add_argument('--handoff', action='store_true', help='After collection completes, stop only the shared-GSM worker so the old GPU can run original happy.')
    parser.add_argument('--destination', type=Path, default=Path('/workspace'))
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    while not complete():
        if not args.wait:
            raise RuntimeError('Shared GSM happy collection is not complete yet. Add --wait; no partial vectors will be exported.')
        print('Waiting for completed shared GSM angry AND happy vector/collection receipts...', flush=True)
        time.sleep(30)
    receipt = environment()
    if len(receipt['hf_model_revisions']) != 3:
        raise RuntimeError('Could not identify all three cached source-model revisions. Inspect HF_HOME/HF_HUB_CACHE before exporting; no floating revision handoff.')
    revisions = dict(models=receipt['hf_model_revisions'])
    receipt_path = repo/'outputs/diffa-model-revisions.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != revisions:
        raise ValueError('Cached source-model revisions changed since the original receipt; inspect before export.')
    receipt_path.write_text(json.dumps(revisions, indent=2)+'\n')
    if args.handoff:
        stop_only_shared_gsm_worker(repo)
    files, missing = files_for(repo)
    # Do not transfer original-method vectors, old pair caches or unfinished aggregate states.
    selected = select_handoff(files, repo)
    selected[receipt_path] = snapshot(receipt_path)
    for root_name in ('diffa-shared-three-tasks',):
        root = repo/'outputs'/root_name
        if root.exists():
            for path in root.rglob('*'):
                if path.is_file() and (path.suffix in ('.json', '.jsonl', '.csv') or path.name == 'style-vectors.pt') and 'pairs' not in path.parts:
                    selected[path] = snapshot(path)
    for path in (repo/'outputs/shared-three-emotions/classifier').glob('*.json'):
        selected[path] = snapshot(path)
    # Source templates contain the original IDs, prompts, metadata and baseline receipts.
    for name in ('gsm8k', 'svamp', 'multiarith', 'bbh3', 'bbh5', 'ifeval'):
        path = repo/'outputs/diffa-happy-winners'/name/'run_config.json'
        if not path.exists():
            raise FileNotFoundError(f'Required original source template is missing: {path}')
        selected[path] = snapshot(path)
    args.destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    final = args.destination/f'diffa-agreed-runs-{stamp}.zip'
    partial = final.with_suffix('.zip.partial')
    manifest = dict(repo=str(repo), git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                    source_environment=receipt, files=[], missing_optional_roots=missing,
                    scope='Completed NEW agreed vectors plus configs, full saved e2v labels and raw baseline/results. '
                          'No original-method vectors, model weights, audio, per-pair caches or unfinished aggregates.',
                    gsm_split=json.loads((GSM/'split.json').read_text()))
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path, data in sorted(selected.items()):
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
    print(f'DOWNLOAD: {final}\nSIZE: {final.stat().st_size/2**20:.1f} MiB\nCHECKSUM: {final}.sha256', flush=True)
    print('Transfer this alongside the audio ZIP. No additional emotion classification is needed.')


if __name__ == '__main__':
    main()
