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
    'flamingo': ('gsm400-candidates/flamingo', 'gsm8k-candidates/flamingo', 'bbh-three-candidates/flamingo',
                 'azure-kimi-gsm8k1269', 'azure-bbh-three-models-regenerated'),
    'strengths': ('winning-strengths', 'reasoning-transfer/gsm8k', 'reasoning-transfer/svamp',
                  'multiarith580-angry-mean', 'ifeval-angry-mean', 'bbh-logical250-angry-mean',
                  'bbh-five250-angry-mean', 'azure-spoken-ifeval', 'azure-multiarith580',
                  'azure-bbh-logical250-steering', 'azure-bbh-five250'),
    'multiarith': ('multiarith580-angry-mean', 'azure-multiarith580'),
    'ifeval': ('ifeval-angry-mean', 'spoken-ifeval-diffa2', 'azure-spoken-ifeval'),
    'moss': ('bbh-three-regenerated-models/moss', 'bbh-three-audio-models/moss', 'azure-bbh-three-models-regenerated'),
    'kimi': ('bbh-three-regenerated-models/kimi', 'bbh-three-audio-models/kimi', 'azure-bbh-three-models-regenerated'),
    'phi': ('gsm8k-candidates/phi4', 'gsm-happy400/phi4', 'azure-gsm-happy400', 'bbh-three-candidates/phi4', 'bbh-five-candidates/phi4',
            'phi4-answer-only-followup', 'phi4-reasoning-emotion100-gsm8k', 'phi4-gsm100-layer-drift', 'phi4-gsm100-layer-drift-neutral-from-angry',
            'phi4-gsm400-all-layers-test869-minus1',
            'phi4-instruction-controls100', 'phi4-instruction-controls100-format-please',
            'phi4-gsm-reverse400-400-L17', 'phi4-gsm-reverse400-400-L30',
            'phi4-gsm-reverse400-869-L17', 'phi4-gsm-reverse400-869-L30',
            'phi4-gsm-reverse400-869-L30-a-1.5', 'phi4-gsm-reverse400-869-L30-a-2',
            'phi4-gsm-reverse400-869-L17-a-1.5', 'phi4-gsm-reverse400-869-L17-a-2',
            'phi4-multiarith-reverse290-290-L17', 'phi4-multiarith-reverse290-290-L30',
            'phi4-multiarith-reverse290-290-L17-extra-minus1.5-minus2',
            'phi4-multiarith-reverse290-shared', 'azure-kimi-gsm8k1269', 'azure-multiarith580',
            'azure-bbh-three-models-regenerated', 'azure-kimi-bbh-five250'),
    'minicpm': ('minicpm-gsm100-diagnostics', 'minicpm-gsm400-869-L30-a1', 'gsm-happy400/minicpm', 'azure-gsm-happy400',
                'gsm-happy400-start400-n400/minicpm', 'azure-gsm-happy400-start400-n400', 'bbh-three-candidates/minicpm', 'gsm8k-candidates/minicpm',
                'azure-kimi-gsm8k1269', 'azure-bbh-three-models-regenerated'),
}
GROUPS['additional-benchmarks'] = ('bbh-tracking250', 'bbh-reasoning200-diffa2', 'azure-bbh-reasoning200',
    'azure-bbh-reasoning250', 'spoken-ifbench-diffa2', 'azure-spoken-ifbench',
    'bbh-seven250-angry-mean', 'azure-bbh-seven250')
GROUPS['step'] = ('step-gsm400', 'step-gsm718', 'step-calculations20', 'step-gsm400-calculations', 'step-gsm400-vectors')
GROUPS['step-vectors'] = ('step-gsm400-vectors',)
GROUPS['emotions'] = ('available-audio-emotions',)
GROUPS['emotion-steering'] = ('diffa-happy-winners', 'diffa-gsm8k-happy', 'diffa-gsm8k-sad')
GROUPS['angry-agreed'] = ('diffa-angry-agreed',)
GROUPS['happy-agreed'] = ('diffa-gsm-happy-agreed',)
GROUPS['bbh-angry125'] = ('diffa-bbh-angry-only125', 'diffa-angry-agreed/audit')
GROUPS['angry-remaining'] = ('diffa-angry-only-remaining', 'diffa-angry-agreed/audit')
GROUPS['phi'] += ('phi4-L0audio-minus1-L11answer-minus1',)
GROUPS['phi'] += ('phi4-L11-L30-double-triple',)
GROUPS['voice'] = ('multiarith290-guy-delivery', 'azure-multiarith290-guy', 'multiarith580-angry-mean',
                   'multiarith290-guy-four-emotions', 'azure-multiarith290-guy-happy-sad')
GROUPS['transfer'] = ('winning-transfer', *GROUPS['strengths'][1:])
GROUPS['phi'] += ('phi4-svamp-L0audio-minus1-L11answer-minus1', 'phi4-svamp-L0audio-minus2-L30answer-minus2-combined-only', 'phi4-bbh-combined', 'phi4-ifeval-all-layers-minus1', 'phi4-svamp-L0audio-minus1-L30answer-minus1-combined-only', 'phi4-svamp-L0audio-minus1-L30answer-minus2-combined-only', 'phi4-svamp-L0audio-minus1-L30answer-minus2', 'phi4-multiarith-L0audio-minus1-L30answer-minus2', 'phi4-gsm-L0audio-minus1-L30answer-minus2', 'phi4-sweeps', 'phi4-next-tasks', 'phi4-task-steering', 'phi4-gsm400-opposite-all-layers-minus1',
                  'phi4-gsm400-test869-L7-minus1-minus1.5', 'phi4-gsm400-test869-L2-minus1-minus1.5')
EXTRA_RUNS = {f'Phi {name} L{layer}': f'phi4-{folder}-L{layer}'
              for name, folder in [('GSM400', 'gsm-reverse400-400'), ('GSM869', 'gsm-reverse400-869'),
                                   ('MultiArith290', 'multiarith-reverse290-290')] for layer in (17, 30)}
EXTRA_RUNS.update({'Phi GSM original': 'gsm8k-candidates/phi4/original',
                   'Phi BBH3 original': 'bbh-three-candidates/phi4/original',
                   'Phi BBH3 explicit': 'bbh-three-candidates/phi4/explicit',
                   'MiniCPM GSM diagnostics': 'minicpm-gsm100-diagnostics',
                   'MiniCPM GSM steering': 'minicpm-gsm400-869-L30-a1',
                   'MiniCPM GSM happy400': 'gsm-happy400/minicpm',
                   'MiniCPM GSM happy next400': 'gsm-happy400-start400-n400/minicpm',
                   'Phi GSM happy400': 'gsm-happy400/phi4',
                   'Phi GSM L30 -1.5': 'phi4-gsm-reverse400-869-L30-a-1.5',
                   'Phi GSM L30 -2': 'phi4-gsm-reverse400-869-L30-a-2'})
EXTRA_RUNS['Phi MultiArith L17 -1.5/-2'] = 'phi4-multiarith-reverse290-290-L17-extra-minus1.5-minus2'
EXTRA_RUNS['Step disjoint vector collection'] = 'step-gsm400-vectors'
EXTRA_RUNS['Flamingo GSM400'] = 'gsm400-candidates/flamingo/original'
EXTRA_RUNS['Phi GSM869 L7 -1/-1.5'] = 'phi4-gsm400-test869-L7-minus1-minus1.5'
EXTRA_RUNS['Phi opposite recovery diagnostic'] = 'phi4-gsm400-opposite-all-layers-minus1'
EXTRA_RUNS['DIFFA Guy four emotions'] = 'multiarith290-guy-four-emotions'
EXTRA_RUNS['DIFFA MultiArith Guy delivery'] = 'multiarith290-guy-delivery'
EXTRA_RUNS['Phi all-layer recovery diagnostic'] = 'phi4-gsm400-all-layers-test869-minus1'
EXTRA_RUNS.update({f'Phi GSM L17 {a}': f'phi4-gsm-reverse400-869-L17-a{a}' for a in ('-1.5', '-2')})


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
    for name in ('recordings', 'extraction', 'collection', 'test', 'pairs'):
        for row in plan.get(name, []) if isinstance(plan.get(name), list) else []:
            if not isinstance(row, dict):
                continue
            rows.append(row)
            rows.extend(row[s] for s in ('neutral', 'angry') if isinstance(row.get(s), dict))
    for name in ('source', 'target'):
        if isinstance(plan.get(name), dict):
            rows.extend(r for r in plan[name].get('records', []) if isinstance(r, dict))
    return rows


def audio_key(row, name):
    return (row.get('question_id', row.get('clip_id')), row.get('voice', row.get('speaker')),
            row.get('style'), Path(name).name)


def source_audio_index(metadata_files):
    """Resolve evaluation-relative references through matching TTS plan records."""
    index = {}
    for config in metadata_files:
        if config.name != 'plan.jsonl':
            continue
        try:
            data, _ = snapshot(config)
            rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        except (OSError, ValueError, UnicodeDecodeError):
            continue  # The archive inspection reports unreadable plans separately.
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('audio_path'), str):
                continue
            path = Path(row['audio_path'])
            path = path if path.is_absolute() else config.parent/path
            if path.is_file():
                index.setdefault(audio_key(row, path.name), []).append(path)
    return index


def audio_path(row, config, repo, source_index=None):
    choices, indexed = [], []
    for key in ('audio_file', 'audio_path'):
        value = row.get(key)
        if not isinstance(value, str) or Path(value).suffix.lower() not in AUDIO:
            continue
        path = Path(value)
        if path.is_absolute():
            choices.append(path)
        else:
            choices.extend([config.parent/path, repo/path])
        indexed.extend((source_index or {}).get(audio_key(row, path.name), []))
    expected = row.get('audio_sha256')
    def matches(path):
        if not path.is_file(): return False
        return not expected or hashlib.sha256(path.read_bytes()).hexdigest() == expected
    for path in choices:
        if matches(path): return path
    candidates = {p.resolve(): p for p in indexed if matches(p)}
    if candidates:
        # Without a receipt, distinct matching recordings must still be byte-identical.
        if expected or len(candidates) == 1 or len({hashlib.sha256(p.read_bytes()).hexdigest() for p in candidates}) == 1:
            return next(iter(candidates.values()))
        raise ValueError('Multiple matching source recordings have different bytes; cannot resolve audio unambiguously.')
    if any(p.is_file() for p in choices):
        raise ValueError('Existing referenced audio does not match the recorded SHA256.')
    return choices[0] if choices else None


def status(repo):
    result = []
    runs = {**RUNS, **{name: relative for name, relative in EXTRA_RUNS.items() if (repo/'outputs'/relative).exists()}}
    for kind in ('strengths', 'transfer'):
        for config in sorted((repo/'outputs'/f'winning-{kind}').glob('*/run_config.json')):
            runs[f'DIFFA {kind} '+config.parent.name] = str(config.parent.relative_to(repo/'outputs'))
    for config in sorted((repo/'outputs/phi4-task-steering').glob('*/*/run_config.json')):
        runs['Phi own-task '+str(config.parent.relative_to(repo/'outputs/phi4-task-steering'))] = str(config.parent.relative_to(repo/'outputs'))
    for name, relative in runs.items():
        folder = repo/'outputs'/relative
        config = next((p for p in (folder/'run_config.json', folder/'split.json') if p.is_file()), None)
        answers = folder/'results.jsonl'
        answer_paths = [answers]+sorted(folder.glob('results-worker*.jsonl'))
        item = dict(experiment=name, directory=str(folder), config_present=config is not None,
                    answers_present=any(p.is_file() for p in answer_paths), vector_present=(folder/'style-vectors.pt').is_file())
        if config:
            try:
                plan = json.loads(snapshot(config)[0])
                item['planned_answers'] = (len(plan['cases'])*len(plan['layers'])*len(plan['groups'])*len(plan['input_styles']) if plan.get('evaluation_kind') == 'opposite-recovery-diagnostic' else
                    len(plan['recordings']) if 'recordings' in plan else
                    len(plan['strengths'])*len(plan['target']['records']) if plan.get('kind') in ('strengths', 'transfer') else
                    len(plan['conditions'])*len(plan['pairs']) if plan.get('protocol', '').startswith('minicpm-gsm-same100') else
                    len(plan.get('strengths', {0: 0, 1: 1, 2: 2}))*len(plan['test']))
                if plan.get('evaluation_kind') == 'all-layer-diagnostic':
                    challenge = folder/'challenge.json'
                    item['planned_answers'] = (len(json.loads(snapshot(challenge)[0])['cases'])*len(plan['layers'])*len(plan['groups'])
                        if challenge.exists() else None)
                    item['planned_reference_answers'] = len(plan['recordings'])
                vector = (plan['source'].get('vector') if isinstance(plan.get('source'), dict) else None) or plan.get('vector') or plan.get('settings', {}).get('vectors')
                if vector:
                    vector_path = Path(vector)
                    if not vector_path.is_absolute():
                        vector_path = repo/vector_path
                    item['required_vector'] = str(vector_path)
                    item['required_vector_present'] = vector_path.is_file()
                elif 'extraction' in plan or plan.get('protocol', '').startswith(('phi4-', 'minicpm-gsm')):
                    item['required_vector'] = str(folder/'style-vectors.pt')
                    item['required_vector_present'] = item['vector_present']
            except (KeyError, ValueError, TypeError):
                item['planned_answers'] = None
        if item['answers_present']:
            try:
                rows = [json.loads(line) for p in answer_paths if p.is_file() for line in snapshot(p)[0].splitlines() if line.strip()]
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
    preserve_small_pairs = bool(only and any(g in only for g in ('phi', 'minicpm')))
    if preserve_small_pairs:
        manifest['note'] = 'Live read-only snapshot. Selected Phi/MiniCPM small per-question mean/state caches included; model weights excluded. Incomplete final JSONL fragments omitted in archive only.'
    preserve_step_pairs=bool(only and 'step-vectors' in only)
    if preserve_step_pairs:manifest['note']='Live read-only snapshot; Step per-pair means up to4 MiB included for vector audits, alongside raw answers/configs/vectors; model weights excluded.'
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
                elif file.suffix == '.pt' and preserve_small_pairs and root.name.startswith(('phi4', 'minicpm')) and file.stat().st_size <= 2*1024**2:
                    metadata_files.append(file)
                elif file.suffix == '.pt' and preserve_step_pairs and root.name=='step-gsm400-vectors' and file.stat().st_size <= 4*1024**2:
                    metadata_files.append(file)
                elif file.suffix == '.pt' and not any('pairs' in p or 'activations' in p for p in file.relative_to(root).parts[:-1]):
                    if 'vector' in file.name or file.stat().st_size <= 64*1024**2:
                        metadata_files.append(file)
    if not (only and 'phi-deliveries' in only):metadata_files.extend(repo.glob('*.log'))
    if preserve_small_pairs:
        if 'phi' in only: metadata_files.extend(outputs.glob('*phi4*.log'))
        if 'minicpm' in only: metadata_files.extend(outputs.glob('*minicpm*.log'))
        for name in ('phi4', 'minicpm'):
            requirements = repo/f'.venv-bbh-{name}'/'experiment-requirements.txt'
            if requirements.is_file(): metadata_files.append(requirements)
    if only and set(only) & {'strengths', 'transfer', 'voice'}:
        for pattern in ('*strength*.log', '*transfer*.log', '*guy*.log'):
            metadata_files.extend(outputs.glob(pattern))
        requirements = repo/'.venv-diffa/experiment-requirements.txt'
        if requirements.is_file(): metadata_files.append(requirements)
    if only and 'flamingo' in only:
        metadata_files.extend(outputs.glob('*flamingo*.log'))
        requirements = repo/'.venv-bbh-flamingo/experiment-requirements.txt'
        if requirements.is_file(): metadata_files.append(requirements)
    if only and 'emotions' in only:
        metadata_files.extend(outputs.glob('*emotion*.log'))
        receipt = outputs/'available-audio-emotions/inventory_manifest.json'
        if receipt.is_file():
            for item in json.loads(snapshot(receipt)[0]).get('legacy_predictions', []):
                file = Path(item['path'])
                if outputs in file.parents and file.suffix.lower() in TEXT and file.is_file():
                    metadata_files.append(file)
    source_index = source_audio_index(metadata_files)
    tracked = subprocess.run(['git', 'ls-files', '-z'], cwd=repo, capture_output=True, check=True).stdout
    code_files = [repo/os.fsdecode(p) for p in tracked.split(b'\0') if p]
    scorer = repo/'external/ifeval'
    if scorer.exists():
        code_files.extend(p for p in walk(scorer) if p.suffix in {'.py', '.jsonl', '.txt'})
    if (repo/'RECOVERY_PLAN.md').is_file():
        code_files.append(repo/'RECOVERY_PLAN.md')

    if only and 'additional-benchmarks' in only:
        upstream=repo/'external/ifbench'
        if upstream.is_dir():
            code_files.extend(p for p in walk(upstream) if p.suffix in {'.py','.jsonl','.txt'} or p.name=='LICENSE')

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
            if data is None or file.name not in ('plan.jsonl', 'run_config.json', 'split.json', 'collection-config.json', 'data_config.json', 'audio_inventory.jsonl'):
                continue
            try:
                plan = [json.loads(line) for line in data.splitlines() if line.strip()] if file.suffix == '.jsonl' else json.loads(data)
                rows = candidates(plan)
            except (ValueError, TypeError):
                manifest['issues'].append(dict(path=str(file), error='Could not inspect audio references in this snapshot.'))
                continue
            selected = []
            sections = [plan] if isinstance(plan, list) else [plan.get(k, []) for k in ('recordings', 'extraction', 'collection', 'test', 'pairs')]
            if isinstance(plan, dict):
                sections.extend(plan[k].get('records', []) for k in ('source', 'target') if isinstance(plan.get(k), dict))
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
                try:
                    audio = audio_path(row, file, repo, source_index)
                except (OSError, ValueError) as error:
                    if row.get('question_id', row.get('clip_id')) in selected:
                        manifest['issues'].append(dict(path=str(file), question_id=row.get('question_id'), error=str(error)))
                        missing_groups.add(row.get('question_id', row.get('clip_id')))
                    continue
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
