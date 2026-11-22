"""ZIP only validated six-dataset Jenny audio, its source plans and exact-hash labels."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from check_sad_audio import rows


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path, default=Path('outputs/audio-transfer-check'))
    parser.add_argument('--destination', type=Path, default=Path('/workspace'))
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    summary = json.loads((args.check/'summary.json').read_text())
    if summary['errors']:
        raise ValueError('Resolve integrity errors before packaging.')
    clips = list(rows(args.check/'validated-audio.jsonl'))
    if len(clips) != summary['valid_unique']:
        raise ValueError('Validated inventory differs from the integrity report; rerun the check.')
    sources = {}
    metadata = set()
    for clip in clips:
        key = clip['audio_sha256']
        paths = [Path(clip['audio_file']), *map(Path, clip.get('paths', []))]
        for path in paths:
            if not path.is_file():
                continue
            logical = path.absolute()
            try:
                member = logical.relative_to(repo).as_posix()
            except ValueError:
                member = f'external/{key}/{path.name}'
            previous = sources.get(member)
            if previous and previous[1] != key:
                raise ValueError(f'Conflicting archive path: {member}')
            sources[member] = (path, key)
        for alias in clip.get('aliases', []):
            if alias.get('source_config'):
                plan = Path(alias['source_config'])
                if plan.name == 'plan.jsonl' and plan.is_file():
                    metadata.add(plan)
                    questions = plan.parent/'questions.jsonl'
                    if questions.is_file():
                        metadata.add(questions)
    labels = {}
    keys = {r['audio_sha256'] for r in clips}
    for path in sorted(Path('outputs').rglob('results-emotion2vec.jsonl')):
        for row in rows(path):
            key = row.get('audio_sha256') or row.get('clip_id')
            if (key in keys and row.get('model') == 'emotion2vec/emotion2vec_plus_large'
                    and row.get('processing_mode') == 'whole_recording'):
                if key not in labels or row.get('classification_status') == 'classified':
                    labels[key] = row
    args.destination.mkdir(parents=True, exist_ok=True)
    raw = sum(path.stat().st_size for path, _ in sources.values())
    free = shutil.disk_usage(args.destination).free
    print(f'{len(clips):,} unique recordings; {len(sources):,} original audio paths; '
          f'{raw/2**30:.2f} GiB before ZIP compression.', flush=True)
    print(f'Free destination space: {free/2**30:.2f} GiB. No audio is converted or deleted.', flush=True)
    if free < 512 * 1024**2:
        raise RuntimeError('Need more than 512 MiB free to start the archive.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    final = args.destination/f'jenny-six-datasets-audio-{stamp}.zip'
    partial = final.with_suffix('.zip.partial')
    manifest = dict(scope='Jenny neutral/angry/happy/sad, six experiment datasets; all classifier outcomes.',
                    integrity_summary=summary, files=[], labels=len(labels),
                    note='Original audio bytes and repository-relative paths retained. External paths are mapped explicitly. '
                         'No model caches, credentials, or generated model answers included.')
    verified = set()
    try:
        with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for index, (member, (path, key)) in enumerate(sorted(sources.items()), 1):
                if shutil.disk_usage(args.destination).free < 256 * 1024**2:
                    raise RuntimeError('Stopped before filling the disk; partial ZIP removed. Choose another --destination.')
                identity = (str(path.resolve()), key)
                if identity not in verified:
                    if sha(path) != key:
                        raise ValueError(f'Audio changed since the integrity check: {path}')
                    verified.add(identity)
                archive.write(path, member)
                manifest['files'].append(dict(source=str(path), member=member, audio_sha256=key))
                if index % 1000 == 0:
                    print(f'Packed {index:,}/{len(sources):,} audio paths; ZIP so far {partial.stat().st_size/2**30:.2f} GiB.', flush=True)
            for path in sorted(metadata):
                try:
                    member = path.absolute().relative_to(repo).as_posix()
                except ValueError:
                    member = f'metadata/external/{sha(path)}/{path.name}'
                archive.write(path, member)
            archive.writestr('JENNY_AUDIO_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
            archive.writestr('metadata/audio_inventory.jsonl', ''.join(json.dumps(r)+'\n' for r in clips))
            archive.writestr('metadata/results-emotion2vec.jsonl', ''.join(json.dumps(r)+'\n' for r in labels.values()))
            for source, member in [(args.check/'summary.json', 'metadata/integrity-summary.json'),
                                   (Path('outputs/shared-three-emotions/shared-question-counts.json'), 'metadata/shared-question-counts.json')]:
                if source.is_file():
                    archive.write(source, member)
        print('Verifying ZIP contents...', flush=True)
        with zipfile.ZipFile(partial) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f'ZIP verification failed: {bad}')
        partial.rename(final)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    checksum = final.with_suffix('.zip.sha256')
    checksum.write_text(f'{sha(final)}  {final.name}\n')
    print(f'DOWNLOAD: {final}\nSIZE: {final.stat().st_size/2**30:.3f} GiB\nCHECKSUM: {checksum}', flush=True)


if __name__ == '__main__':
    main()
