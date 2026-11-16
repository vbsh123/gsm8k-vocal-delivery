"""Read-only evidence ZIP outside outputs: results, configs, vectors, code and matched audio samples."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import zipfile

TEXT = {'.json', '.jsonl', '.yaml', '.yml', '.csv', '.tsv', '.txt', '.md', '.html', '.log', '.py', '.sh'}
AUDIO = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
SKIP = {'.git', '.cache', '__pycache__', 'hub', 'blobs', 'snapshots', 'node_modules'}
RUNS = {
    'GSM8K aligned': 'reasoning-transfer/gsm8k',
    'SVAMP steering': 'reasoning-transfer/svamp',
    'BBH three steering': 'bbh-logical250-angry-mean',
    'BBH five own vector': 'bbh-five250-angry-mean',
    'BBH seven steering': 'bbh-seven250-angry-mean',
    'IFEval steering': 'ifeval-angry-mean',
    'MultiArith steering': 'multiarith580-angry-mean',
    'Tracking steering': 'bbh-tracking250/steering',
    'IFBench delivery': 'spoken-ifbench-diffa2',
    'MOSS regenerated': 'bbh-three-regenerated-models/moss',
    'Kimi regenerated': 'bbh-three-regenerated-models/kimi',
}
GROUPS = {
    'multiarith': ('multiarith580-angry-mean', 'azure-multiarith580'),
    'ifeval': ('ifeval-angry-mean', 'spoken-ifeval-diffa2', 'azure-spoken-ifeval'),
    'moss': ('bbh-three-regenerated-models/moss', 'bbh-three-audio-models/moss', 'azure-bbh-three-models-regenerated'),
    'kimi': ('bbh-three-regenerated-models/kimi', 'bbh-three-audio-models/kimi', 'azure-bbh-three-models-regenerated'),
}


def walk(root):
    for directory, names, files in os.walk(root):
        names[:] = sorted(n for n in names if n not in SKIP and not n.startswith(('hf-', 'models--', 'datasets--', '.venv')))
        for name in sorted(files):
            yield Path(directory)/name


def snapshot(path):
    data = path.read_bytes()
    trimmed = False
    if path.suffix == '.jsonl' and data and not data.endswith(b'\n'):
        tail = data[data.rfind(b'\n')+1:]
        try:
            json.loads(tail)
        except (ValueError, UnicodeDecodeError):
            data = data[:data.rfind(b'\n')+1]
            trimmed = True
    return data, trimmed


def candidates(plan):
    if isinstance(plan, list):
        return [r for r in plan if isinstance(r, dict)]
    rows = []
    for name in ('recordings', 'extraction', 'test', 'pairs'):
        for row in plan.get(name, []) if isinstance(plan.get(name), list) else []:
            if not isinstance(row, dict):
                continue
            rows.append(row)
            rows.extend(row[s] for s in ('neutral', 'angry') if isinstance(row.get(s), dict))
    return rows


def audio_path(row, config, repo):
    for key in ('audio_file', 'audio_path'):
        value = row.get(key)
        if not isinstance(value, str) or Path(value).suffix.lower() not in AUDIO:
            continue
        path = Path(value)
        if path.is_absolute():
            return path
        choices = [config.parent/path, repo/path]
        return next((p for p in choices if p.is_file()), choices[0])
    return None


def status(repo):
    result = []
    for name, relative in RUNS.items():
        folder = repo/'outputs'/relative
        config = next((p for p in (folder/'run_config.json', folder/'split.json') if p.is_file()), None)
        answers = folder/'results.jsonl'
        item = dict(experiment=name, directory=str(folder), config_present=config is not None,
                    answers_present=answers.is_file(), vector_present=(folder/'style-vectors.pt').is_file())
        if config:
            try:
                plan = json.loads(snapshot(config)[0])
                item['planned_answers'] = len(plan['recordings']) if 'recordings' in plan else 3*len(plan['test'])
                vector = plan.get('vector') or plan.get('settings', {}).get('vectors')
                if vector:
                    vector_path = Path(vector)
                    if not vector_path.is_absolute():
                        vector_path = repo/vector_path
                    item['required_vector'] = str(vector_path)
                    item['required_vector_present'] = vector_path.is_file()
                elif 'extraction' in plan:
                    item['required_vector'] = str(folder/'style-vectors.pt')
                    item['required_vector_present'] = item['vector_present']
            except (KeyError, ValueError, TypeError):
                item['planned_answers'] = None
        if answers.is_file():
            try:
                rows = [json.loads(line) for line in snapshot(answers)[0].splitlines() if line.strip()]
                keys = {(r.get('question_id', r.get('clip_id')), r.get('voice', r.get('speaker')),
                         r.get('condition', r.get('style'))) for r in rows}
                item['saved_unique_answers'] = len(keys)
                item['conditions'] = dict(Counter(r.get('condition', r.get('style', 'unknown')) for r in rows))
            except (ValueError, TypeError):
                item['answer_parse_error'] = True
        result.append(item)
    return result


def package(repo, destination, samples=3, sample_mb=256, full_audio=False, only=None):
    repo, destination = repo.resolve(), destination.resolve()
    outputs = repo/'outputs'
    if destination == outputs or outputs in destination.parents:
        raise ValueError('Archive destination must be outside outputs.')
    destination.mkdir(parents=True, exist_ok=True)
    host = ''.join(c if c.isalnum() or c in '-_' else '-' for c in socket.gethostname())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    label = '-'+ '-'.join(only) if only else ''
    path = destination/f'experiment-evidence{label}-{host}-{stamp}.zip'
    partial = path.with_suffix('.zip.partial')
    manifest = dict(created_utc=stamp, host=host, repo=str(repo), files=[], issues=[], audio_samples=[],
        runs=status(repo), note='Live read-only snapshot; incomplete final JSONL fragments omitted in archive only. Pair activation caches/model weights excluded.')
    roots = [outputs, repo/'recovery-backups']
    if only:
        roots = sorted({root/name for root in roots for group in only for name in GROUPS[group]})
        manifest['only_groups'] = list(only)
        manifest['runs'] = [r for r in manifest['runs'] if any(
            Path(r['directory']) == root or root in Path(r['directory']).parents for root in roots)]
    audio_files = {}; seen_samples = set(); sampled_bytes = 0
    metadata_files = []
    for root in roots:
        if root.exists():
            for file in walk(root):
                if file.suffix.lower() in TEXT:
                    metadata_files.append(file)
                elif file.suffix == '.pt' and not any('pairs' in p or 'activations' in p for p in file.relative_to(root).parts[:-1]):
                    if 'vector' in file.name or file.stat().st_size <= 64*1024**2:
                        metadata_files.append(file)
    metadata_files.extend(repo.glob('*.log'))
    tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=repo, capture_output=True, check=True).stdout
    code_files = [repo/os.fsdecode(p) for p in tracked.split(b'\0') if p]
    scorer = repo/'external/ifeval'
    if scorer.exists():
        code_files.extend(p for p in walk(scorer) if p.suffix in {'.py', '.jsonl', '.txt'})
    if (repo/'RECOVERY_PLAN.md').is_file():
        code_files.append(repo/'RECOVERY_PLAN.md')

    packed_sources = set()
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        def add(file, member):
            try:
                data, trimmed = snapshot(file)
                archive.writestr(member, data)
                manifest['files'].append(dict(member=member, source=str(file), bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(), incomplete_jsonl_tail_omitted=trimmed))
                packed_sources.add(file.resolve())
                return data
            except OSError as error:
                manifest['issues'].append(dict(path=str(file), error=str(error)))
                return None

        for file in sorted(set(code_files)):
            if file.is_file():
                add(file, 'code/'+file.relative_to(repo).as_posix())
        core_roots = [outputs/relative for relative in RUNS.values()]
        for file in sorted(set(metadata_files), key=lambda p: (not any(root in p.parents for root in core_roots), str(p))):
            data = add(file, file.relative_to(repo).as_posix())
            if data is None or file.name not in ('plan.jsonl', 'run_config.json', 'split.json', 'collection-config.json'):
                continue
            try:
                plan = [json.loads(line) for line in data.splitlines() if line.strip()] if file.suffix == '.jsonl' else json.loads(data)
                rows = candidates(plan)
            except (ValueError, TypeError):
                manifest['issues'].append(dict(path=str(file), error='Could not inspect audio references in this snapshot.'))
                continue
            selected = []
            sections = [plan] if isinstance(plan, list) else [plan.get(k, []) for k in ('recordings', 'extraction', 'test', 'pairs')]
            for section in sections:
                ids = []
                for row in section if isinstance(section, list) else []:
                    if not isinstance(row, dict):
                        continue
                    qid = row.get('question_id', row.get('clip_id'))
                    if qid is not None and qid not in ids and len(ids) < samples:
                        ids.append(qid)
                selected.extend(qid for qid in ids if qid not in selected)
            groups = {}; missing_groups = set()
            for row in rows:
                audio = audio_path(row, file, repo)
                if audio is None:
                    continue
                if not audio.is_file():
                    if row.get('question_id', row.get('clip_id')) in selected:
                        manifest['issues'].append(dict(path=str(audio), error='Referenced sample audio missing.'))
                        missing_groups.add(row.get('question_id', row.get('clip_id')))
                    continue
                real = audio.resolve()
                if file.name == 'plan.jsonl' and row.get('provider') in ('azure', 'elevenlabs'):
                    audio_files.setdefault(audio.absolute(), real)
                if row.get('question_id', row.get('clip_id')) not in selected or real in seen_samples:
                    continue
                groups.setdefault(row.get('question_id', row.get('clip_id')), {})[real] = (audio, row)
            for qid, clips in groups.items():
                if qid in missing_groups:
                    continue
                if sampled_bytes+sum(audio.stat().st_size for audio, row in clips.values()) > sample_mb*1024**2:
                    continue
                for real, (audio, row) in clips.items():
                    try:
                        relative = real.relative_to(repo).as_posix()
                    except ValueError:
                        relative = 'external/'+hashlib.sha256(str(real).encode()).hexdigest()[:12]+'-'+audio.name
                    member = 'audio-samples/'+relative
                    if add(audio, member) is not None:
                        sampled_bytes += audio.stat().st_size; seen_samples.add(real)
                        manifest['audio_samples'].append(dict(question_id=qid,
                            style=row.get('style'), member=member, referenced_by=str(file)))
        for item in manifest['runs']:
            if item.get('required_vector_present'):
                vector = Path(item['required_vector'])
                if vector.resolve() not in packed_sources:
                    member = 'referenced-vectors/'+hashlib.sha256(str(vector).encode()).hexdigest()[:12]+'-'+vector.name
                    add(vector, member)
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'diff', 'HEAD', '--binary'], cwd=repo, capture_output=True, check=True).stdout
        archive.writestr('metadata/git-commit.txt', head+'\n')
        archive.writestr('metadata/working-tree.patch', dirty)
        archive.writestr('metadata/environment.json', json.dumps(dict(python=sys.version, executable=sys.executable,
            platform=platform.platform(), packages=sorted({(d.metadata.get('Name', ''), d.version) for d in distributions()})), indent=2))
        archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    verify(partial)
    partial.rename(path)
    checksum(path)
    print('\nDOWNLOAD:', path, flush=True)
    print(f'{len(manifest["files"])} files; {len(seen_samples)} audio samples; {path.stat().st_size/1024**2:.1f} MiB.', flush=True)
    print('\nRun evidence on this machine:', flush=True)
    for item in manifest['runs']:
        state = f'{item.get("saved_unique_answers", 0)}/{item.get("planned_answers", "?")} answers'
        missing = ', '.join(label for key, label in [('config_present', 'config'), ('answers_present', 'raw answers')] if not item[key])
        print(f'  {item["experiment"]}: {state}'+(f'; MISSING {missing}' if missing else ''), flush=True)
        if item.get('required_vector_present') is False:
            print('    MISSING required vector:', item['required_vector'], flush=True)
    issues = list({(i['path'], i['error']) for i in manifest['issues']})
    if issues:
        print(f'\nATTENTION: {len(issues)} missing/unreadable artifacts or audio references (full list in MANIFEST.json):', flush=True)
        for file, error in sorted(issues)[:20]:
            print(f'  {file}: {error}', flush=True)
    if full_audio:
        audio_zip = path.with_name(path.stem+'-full-tts-audio.zip')
        print(f'\nPacking {len(audio_files)} original TTS clips; this may be substantially larger.', flush=True)
        with zipfile.ZipFile(audio_zip.with_suffix('.zip.partial'), 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for logical, real in sorted(audio_files.items()):
                try:
                    member = logical.relative_to(repo).as_posix()
                except ValueError:
                    member = 'external/'+hashlib.sha256(str(real).encode()).hexdigest()[:12]+'-'+logical.name
                archive.write(real, member)
        verify(audio_zip.with_suffix('.zip.partial'))
        audio_zip.with_suffix('.zip.partial').rename(audio_zip)
        checksum(audio_zip)
        print('DOWNLOAD FULL AUDIO:', audio_zip, flush=True)
    return path


def verify(path):
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'Archive CRC verification failed: {bad}')


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    line = f'{digest.hexdigest()}  {path.name}\n'
    path.with_suffix(path.suffix+'.sha256').write_text(line)
    print('SHA256:', line.strip(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, default=Path('..'), help='Outside outputs; defaults to /workspace when run from the usual repo')
    p.add_argument('--samples', type=int, default=3, help='First N question groups per audio plan/config, preserving available paired styles')
    p.add_argument('--sample-mb', type=int, default=256, help='Maximum total sampled-audio size before compression')
    p.add_argument('--full-audio', action='store_true', help='Also create a separate ZIP of all available original TTS-plan audio')
    p.add_argument('--only', nargs='+', choices=GROUPS, help='Focus data/config/vector folders on selected recent runs; still include the source snapshot and root logs')
    args = p.parse_args()
    if args.samples < 0 or args.sample_mb < 0:
        p.error('Sample limits must be nonnegative.')
    package(Path.cwd(), args.destination, args.samples, args.sample_mb, args.full_audio, args.only)


if __name__ == '__main__':
    main()
