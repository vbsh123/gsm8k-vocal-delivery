"""Fill missing Jenny emotion/neutral audio; default is a read-only preview."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import socket

from check_sad_audio import PREFIXES as OTHER_PREFIXES, rows

VOICE = 'en-US-JennyNeural'
PREFIXES = {'gsm8k': 'gsm8k_test_', **OTHER_PREFIXES}


def inventory(root):
    bank = {name: {} for name in PREFIXES}
    available = set()
    neutral_texts = {}

    def add(q, text, answer, style, voice, paths):
        if (voice != VOICE or not isinstance(q, str) or not isinstance(text, str)
                or style not in ('neutral', 'angry', 'happy', 'sad')):
            return
        name = next((n for n, prefix in PREFIXES.items() if q.startswith(prefix)), None)
        if name is None:
            return
        if style == 'neutral':
            old_text = neutral_texts.get(q)
            if old_text is not None and old_text != text:
                raise ValueError(f'Differing neutral question texts for {q}; resolve before synthesis.')
            neutral_texts[q] = text
        # Emotion plans also supply references when a neutral recording never existed.
        # Metadata-only aliases without answers must not erase the original reference.
        if answer is not None and (q not in bank[name] or style == 'neutral'):
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
    parser.add_argument('--datasets', nargs='+', choices=list(PREFIXES), default=list(OTHER_PREFIXES))
    parser.add_argument('--styles', nargs='+', choices=['neutral', 'angry', 'sad', 'happy'], default=['sad', 'happy'])
    parser.add_argument('--run', action='store_true', help='Make billable Azure synthesis requests.')
    parser.add_argument('--diagnose', action='store_true', help='Show actual batch files and matching details; never synthesize.')
    args = parser.parse_args()
    if args.diagnose:
        print(f'Machine: {socket.gethostname()}\nWorking directory: {Path.cwd()}\nScan root: {args.root.resolve()}\nGenerated batches: {args.output.resolve()}', flush=True)
        for name in args.datasets:
            for style in args.styles:
                folder = args.output / name / style
                plan = folder / 'plan.jsonl'
                if not plan.exists():
                    print(f'{folder}: no plan.jsonl; WAV files={len(list(folder.glob("*.wav")))}', flush=True)
                    continue
                records = list(rows(plan))
                present = [r for r in records if r.get('audio_path') and (folder / r['audio_path']).is_file()]
                valid = [r for r in present if (folder / r['audio_path']).stat().st_size > 44]
                print(f'{folder}: planned={len(records)}, files={len(present)}, files>44bytes={len(valid)}', flush=True)
                if records:
                    row = records[0]
                    path = folder / row.get('audio_path', 'MISSING_AUDIO_PATH')
                    print(f'  Sample: id={row.get("question_id")!r}, provider={row.get("provider")!r}, voice={row.get("voice")!r}, style={row.get("style")!r}\n  File: {path.resolve()} exists={path.is_file()}', flush=True)
    bank, available = inventory(args.root)
    jobs = []
    total_chars = 0
    for name in dict.fromkeys(args.datasets):
        if not bank[name]:
            raise ValueError(f'{name}: no question bank with references found; nothing generated.')
        for style in dict.fromkeys(args.styles):
            missing = [row for q, row in sorted(bank[name].items())
                       if (q, style, row['question']) not in available]
            print(f'{name:12} {style:6}: {len(missing)}/{len(bank[name])} missing', flush=True)
            if args.diagnose and missing:
                first = missing[0]
                variants = [(s, text == first['question']) for q, s, text in available
                            if q == first['question_id']]
                print(f'  First missing: {first["question_id"]}; available (style, same text): {variants}', flush=True)
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
    if not args.run or args.diagnose:
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
