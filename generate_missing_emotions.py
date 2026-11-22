"""Fill missing Jenny sad/happy audio; default is a read-only preview."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from check_sad_audio import PREFIXES, rows

VOICE = 'en-US-JennyNeural'


def inventory(root):
    bank = {name: {} for name in PREFIXES}
    available = set()

    def add(q, text, answer, style, voice, paths):
        if voice != VOICE or not isinstance(q, str) or not isinstance(text, str):
            return
        name = next((n for n, prefix in PREFIXES.items() if q.startswith(prefix)), None)
        if name is None:
            return
        if style == 'neutral':
            old = bank[name].get(q)
            if old and old['question'] != text:
                raise ValueError(f'Differing neutral question texts for {q}; resolve before synthesis.')
            bank[name][q] = dict(question_id=q, question=text, answer=answer)
        if any(p.is_file() and p.stat().st_size > 44 for p in paths):
            available.add((q, style, text))

    for path in sorted(root.rglob('plan.jsonl')):
        for row in rows(path):
            if row.get('provider') != 'azure':
                continue
            paths = [path.parent / row['audio_path']] if row.get('audio_path') else []
            add(row.get('question_id'), row.get('question'), row.get('answer'),
                row.get('style'), row.get('voice'), paths)
    for path in sorted(root.rglob('audio_inventory.jsonl')):
        for row in rows(path):
            paths = [Path(p) for p in row.get('paths', [])]
            if row.get('audio_file'):
                paths.append(Path(row['audio_file']))
            for alias in row.get('aliases', []):
                add(alias.get('question_id'), alias.get('question'), alias.get('answer'),
                    alias.get('requested_style'), alias.get('voice'), paths)
    return bank, available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs'))
    parser.add_argument('--output', type=Path, default=Path('outputs/missing-emotion-audio'))
    parser.add_argument('--datasets', nargs='+', choices=list(PREFIXES), default=list(PREFIXES))
    parser.add_argument('--styles', nargs='+', choices=['sad', 'happy'], default=['sad', 'happy'])
    parser.add_argument('--run', action='store_true', help='Make billable Azure synthesis requests.')
    args = parser.parse_args()
    bank, available = inventory(args.root)
    jobs = []
    total_chars = 0
    for name in dict.fromkeys(args.datasets):
        if not bank[name]:
            raise ValueError(f'{name}: no neutral question bank found; nothing generated.')
        for style in dict.fromkeys(args.styles):
            missing = [row for q, row in sorted(bank[name].items())
                       if (q, style, row['question']) not in available]
            print(f'{name:12} {style:6}: {len(missing)}/{len(bank[name])} missing', flush=True)
            folder = args.output / name / style
            frozen = folder / 'questions.jsonl'
            questions = list(rows(frozen)) if frozen.exists() else missing
            for row in questions:
                original = bank[name].get(row['question_id'])
                if original is None or original['question'] != row['question']:
                    raise ValueError(f'Frozen question changed: {frozen}')
            if not missing:
                continue
            if not {r['question_id'] for r in missing} <= {r['question_id'] for r in questions}:
                raise ValueError(f'New missing IDs outside frozen batch: {folder}; use another --output.')
            total_chars += sum(len(r['question']) for r in missing)
            jobs.append((folder, questions, style))
    print(f'Missing question text: {total_chars:,} characters. No model inference or classification.', flush=True)
    if not args.run:
        print('Preview only. Add --run to generate missing audio using Azure credentials.')
        return
    if jobs and any(not os.environ.get(k) for k in ('AZURE_SPEECH_KEY', 'AZURE_SPEECH_REGION')):
        raise ValueError('Export AZURE_SPEECH_KEY and AZURE_SPEECH_REGION before running; no API requests made.')
    for folder, questions, style in jobs:
        folder.mkdir(parents=True, exist_ok=True)
        frozen = folder / 'questions.jsonl'
        if not frozen.exists():
            frozen.write_text(''.join(json.dumps(r) + '\n' for r in questions))
        subprocess.run([sys.executable, '-u', 'cloud_tts_smoke.py', '--provider', 'azure',
                        '--azure-voice', VOICE, '--azure-intensity', '1.5',
                        '--input-jsonl', str(frozen), '--num-examples', str(len(questions)),
                        '--styles', style, '--output', str(folder), '--resume',
                        '--retry-passes', '5'], check=True)


if __name__ == '__main__':
    main()
