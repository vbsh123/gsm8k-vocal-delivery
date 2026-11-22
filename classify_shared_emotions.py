"""Classify missing Jenny emotion labels and count same-question angry/happy/sad agreement."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from check_sad_audio import rows
from shared_emotion_counts import MODEL, PREFIXES, count

STYLES = ('neutral', 'angry', 'happy', 'sad')


def write_rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(r)+'\n' for r in records))
    temporary.replace(path)


def prepare(root, output):
    clips = {}
    hashes = {}

    def add(alias, paths):
        q = alias.get('question_id')
        if (alias.get('voice') != 'en-US-JennyNeural'
                or alias.get('requested_style') not in STYLES
                or not isinstance(q, str) or not any(q.startswith(p) for p in PREFIXES.values())):
            return
        for path in paths:
            if not path.is_file() or path.stat().st_size <= 44:
                continue
            path = path.resolve()
            if path not in hashes:
                digest = hashlib.sha256()
                with path.open('rb') as audio:
                    for block in iter(lambda: audio.read(1024 * 1024), b''):
                        digest.update(block)
                hashes[path] = digest.hexdigest()
            key = hashes[path]
            clip = clips.setdefault(key, dict(clip_id=key, audio_sha256=key,
                                               audio_file=str(path), paths=[], aliases=[]))
            if str(path) not in clip['paths']:
                clip['paths'].append(str(path))
            if alias not in clip['aliases']:
                clip['aliases'].append(alias)

    for source in sorted(root.rglob('plan.jsonl')):
        for row in rows(source):
            if row.get('provider') == 'azure' and row.get('audio_path'):
                add(dict(question_id=row.get('question_id'), requested_style=row.get('style'),
                         voice=row.get('voice'), question=row.get('question'), source_config=str(source)),
                    [source.parent / row['audio_path']])
    for source in sorted(root.rglob('audio_inventory.jsonl')):
        for row in rows(source):
            paths = [Path(p) for p in row.get('paths', [])]
            if row.get('audio_file'):
                paths.append(Path(row['audio_file']))
            for alias in row.get('aliases', []):
                add(alias, paths)
    write_rows(output / 'audio_inventory.jsonl', sorted(clips.values(), key=lambda r: r['clip_id']))
    emotions = {k: r for k, r in clips.items()
                if any(a['requested_style'] != 'neutral' for a in r['aliases'])}
    worker = output / 'classifier'
    write_rows(worker / 'audio_inventory.jsonl', sorted(emotions.values(), key=lambda r: r['clip_id']))
    saved = {}
    for source in sorted(root.rglob('results-emotion2vec.jsonl')):
        for row in rows(source):
            key = row.get('audio_sha256') or row.get('clip_id')
            if (key in emotions and row.get('model') == MODEL
                    and row.get('processing_mode') == 'whole_recording'
                    and row.get('classification_status') == 'classified'):
                saved[key] = dict(row, clip_id=key)
    write_rows(worker / 'results-emotion2vec.jsonl', saved.values())
    print(f'{len(emotions)} unique emotional clips; {len(saved)} existing labels reused; '
          f'{len(emotions)-len(saved)} need classification. Neutral is not classified.', flush=True)
    return worker, len(emotions) - len(saved)


def report(root, output):
    cohort_path = root / 'diffa-angry-agreed/audit/cohorts.json'
    cohorts = json.loads(cohort_path.read_text())['datasets'] if cohort_path.exists() else None
    results = count(root, cohorts)
    print('\nSame question IDs: angry AND happy AND sad each recognized by emotion2vec.')
    print('Neutral audio must exist; its prediction is NOT filtered. No response-accuracy filtering.')
    print(f'{"Dataset":12} {"Angry OK":>10} {"Happy OK":>10} {"Sad OK":>10} {"Shared3":>10} {"Outside old test":>18} {"Pending":>10}')
    for name, row in results.items():
        r = row['recognized']
        outside = str(row.get('shared_outside_original_test', 'n/a'))
        print(f'{name:12} {r["angry"]:10} {r["happy"]:10} {r["sad"]:10} '
              f'{row["shared_count"]:10} {outside:>18} {row["unclassified_emotional_clips"]:10}')
    (output / 'shared-question-counts.json').write_text(json.dumps(results, indent=2)+'\n')
    print(f'Counts and shared IDs saved: {output / "shared-question-counts.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['run', 'summary'], default='run')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    root = Path('outputs')
    output = root / 'shared-three-emotions'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == 'run':
            print('Scanning saved audio and reusing exact-hash whole-recording labels.', flush=True)
            worker, pending = prepare(root, output)
            if pending:
                command = [sys.executable, '-u', 'audit_available_audio.py', '--mode', 'classify',
                           '--output', str(worker), '--model', 'emotion2vec', '--device', args.device,
                           '--retry-errors']
                if args.device.startswith('cuda'):
                    with (root / 'jenny-task-vectors.lock').open('a') as gpu:
                        print('Waiting for shared GPU queue for emotion2vec.', flush=True)
                        fcntl.flock(gpu, fcntl.LOCK_EX)
                        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                                      '--format=csv,noheader'], text=True).strip():
                            time.sleep(30)
                        subprocess.run(command, check=True)
                else:
                    subprocess.run(command, check=True)
        report(root, output)


if __name__ == '__main__':
    main()
