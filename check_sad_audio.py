"""Read-only census of surviving Jenny sad audio and saved emotion2vec labels."""
import argparse
import hashlib
import json
from pathlib import Path

PREFIXES = {
    'svamp': 'svamp_chal-', 'multiarith': 'multiarith_',
    'bbh3': 'bbh_logical_three_', 'bbh5': 'bbh_logical_five_',
    'ifeval': 'ifeval_',
}


def rows(path):
    with path.open() as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def census(root):
    found = {name: {} for name in PREFIXES}
    labels = {}

    def add(question, style, voice, paths):
        if style != 'sad' or voice != 'en-US-JennyNeural':
            return
        for name, prefix in PREFIXES.items():
            if isinstance(question, str) and question.startswith(prefix):
                for path in paths:
                    if path.is_file():
                        found[name].setdefault(question, set()).add(path.resolve())

    for source in sorted(root.rglob('results-emotion2vec.jsonl')):
        for row in rows(source):
            if (row.get('model') == 'emotion2vec/emotion2vec_plus_large'
                    and row.get('processing_mode') == 'whole_recording'
                    and row.get('classification_status') == 'classified'):
                key = row.get('audio_sha256') or row.get('clip_id')
                if key:
                    labels[key] = row.get('emotion')
    for source in sorted(root.rglob('audio_inventory.jsonl')):
        for row in rows(source):
            paths = [Path(p) for p in row.get('paths', [])]
            if row.get('audio_file'):
                paths.append(Path(row['audio_file']))
            for alias in row.get('aliases', []):
                add(alias.get('question_id'), alias.get('requested_style'),
                    alias.get('voice'), paths)
    for source in sorted(root.rglob('plan.jsonl')):
        for row in rows(source):
            if row.get('audio_path'):
                add(row.get('question_id'), row.get('style'), row.get('voice'),
                    [source.parent / row['audio_path']])

    hashes = {}
    result = {}
    for name, questions in found.items():
        classified = agreed = 0
        for paths in questions.values():
            predictions = []
            for path in paths:
                if path not in hashes:
                    digest = hashlib.sha256()
                    with path.open('rb') as audio:
                        for chunk in iter(lambda: audio.read(1024 * 1024), b''):
                            digest.update(chunk)
                    hashes[path] = digest.hexdigest()
                if hashes[path] in labels:
                    predictions.append(labels[hashes[path]])
            classified += bool(predictions)
            agreed += 'sad' in predictions
        result[name] = (len(questions), classified, agreed, len(questions) - classified)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    print('Reading saved evidence and hashing sad audio; no inference or GPU.', flush=True)
    result = census(args.root)
    print(f'{"Dataset":12} {"Sad present":>12} {"Classified":>12} {"Sad agreed":>12} {"Unclassified":>14}')
    for name, counts in result.items():
        print(f'{name:12}' + ''.join(f'{value:14}' for value in counts))
    print('Unique question IDs; any labeled rendition counts as classified. Neutral is not filtered.')
    print('No audio found means absent from surviving plans/inventories, not proof that no unindexed file exists.')


if __name__ == '__main__':
    main()
