"""Check the six-dataset Jenny neutral/angry/happy/sad audio before archiving it."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import wave

from check_sad_audio import rows
from shared_emotion_counts import PREFIXES

STYLES = ('neutral', 'angry', 'happy', 'sad')


def validate(path, expected_hash):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected_hash:
        raise ValueError('SHA256 mismatch: file differs from inventoried/classified recording')
    with wave.open(str(path), 'rb') as audio:
        rate, frames = audio.getframerate(), audio.getnframes()
        if rate <= 0 or frames <= 0:
            raise ValueError('Empty/invalid WAV')
        expected_bytes = frames * audio.getnchannels() * audio.getsampwidth()
        actual_bytes = 0
        while True:
            block = audio.readframes(65536)
            if not block:
                break
            actual_bytes += len(block)
        if actual_bytes != expected_bytes:
            raise ValueError(f'Truncated WAV: {actual_bytes}/{expected_bytes} PCM bytes')
        return dict(seconds=frames/rate, sample_rate=rate, channels=audio.getnchannels(),
                    sample_width=audio.getsampwidth(), bytes=path.stat().st_size)


def audit(inventory):
    expected = {n: set() for n in PREFIXES}
    available = {n: {s: set() for s in STYLES} for n in PREFIXES}
    variants = defaultdict(set)
    errors, checked = [], {}
    valid_records = []
    for clip in rows(inventory):
        aliases = []
        for alias in clip.get('aliases', []):
            q = alias.get('question_id') or ''
            name = next((n for n, prefix in PREFIXES.items() if q.startswith(prefix)), None)
            style = alias.get('requested_style')
            if name and style in STYLES and alias.get('voice') == 'en-US-JennyNeural':
                expected[name].add(q)
                aliases.append((name, q, style))
        if not aliases:
            continue
        key = clip['audio_sha256']
        if key not in checked:
            paths = [Path(p) for p in clip.get('paths', [])]
            if clip.get('audio_file'):
                paths.append(Path(clip['audio_file']))
            path = next((p for p in paths if p.is_file()), None)
            try:
                if path is None:
                    raise FileNotFoundError('No surviving path for inventoried audio')
                info = validate(path, key)
                checked[key] = dict(path=str(path.resolve()), **info)
                valid_records.append(dict(clip, audio_file=str(path.resolve())))
            except (OSError, ValueError, wave.Error, EOFError) as exc:
                checked[key] = None
                errors.append(dict(audio_sha256=key, path=str(path), aliases=aliases, error=str(exc)))
        if checked[key] is not None:
            for name, q, style in aliases:
                available[name][style].add(q)
                variants[name, q, style].add(key)
    counts = {}
    for name in PREFIXES:
        groups = available[name]
        counts[name] = dict(question_pool=len(expected[name]),
                            styles={s: len(groups[s]) for s in STYLES},
                            missing={s: sorted(expected[name]-groups[s]) for s in STYLES},
                            complete_four=len(set.intersection(*(groups[s] for s in STYLES))))
    summary = dict(datasets=counts, checked_unique=len(checked),
                   valid_unique=sum(info is not None for info in checked.values()),
                   total_bytes=sum(info['bytes'] for info in checked.values() if info),
                   multiple_renditions=sum(len(keys)>1 for keys in variants.values()), errors=errors,
                   note='Expected pool is the union of question IDs in the saved inventory, not a fresh official-dataset download. '
                        'Counts include all available deliveries regardless of classifier agreement. No audio is rewritten.')
    return summary, valid_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=Path('outputs/shared-three-emotions/audio_inventory.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('outputs/audio-transfer-check'))
    args = parser.parse_args()
    if not args.inventory.is_file():
        raise FileNotFoundError(f'Missing inventory: {args.inventory}')
    print('Checking file hashes and complete WAV payloads; CPU only, no generation/classification.', flush=True)
    summary, valid = audit(args.inventory)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.output/'validated-audio.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in valid))
    print(f'{"Dataset":12} {"Pool":>6} {"Neutral":>9} {"Angry":>9} {"Happy":>9} {"Sad":>9} {"All four":>10}')
    for name, row in summary['datasets'].items():
        print(f'{name:12} {row["question_pool"]:6}' + ''.join(f'{row["styles"][s]:9}' for s in STYLES)
              + f'{row["complete_four"]:10}')
        missing = {s: len(ids) for s, ids in row['missing'].items() if ids}
        if missing:
            print('  Missing/unusable question deliveries:', missing)
    print(f'Valid unique audio: {summary["valid_unique"]}/{summary["checked_unique"]}; '
          f'uncompressed size: {summary["total_bytes"]/2**30:.2f} GiB; errors: {len(summary["errors"])}')
    print(f'Question/style combinations with multiple byte-distinct recordings: {summary["multiple_renditions"]}')
    print(f'Full details: {args.output / "summary.json"}')
    if summary['errors']:
        for error in summary['errors'][:5]:
            print(error['path'], error['error'])
        raise SystemExit(1)


if __name__ == '__main__':
    main()
