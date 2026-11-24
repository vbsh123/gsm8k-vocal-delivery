"""Show three matched AQuA answers, including original and spoken choices; read only."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from steer import read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('outputs/diffa-aqua50'))
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--longest', action='store_true', help='Longest angry outputs; diagnostic, not representative.')
    parser.add_argument('--question-id')
    args = parser.parse_args()
    if args.count < 1: parser.error('--count must be positive')
    path = args.run/'run_config.json'
    if not path.exists(): parser.exit(message='AQuA evaluation not prepared yet; audio may still be generating.\n')
    plan = json.loads(path.read_text())
    if plan.get('name') != 'aqua_rat': parser.error('Expected an AQuA-RAT run')
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = {(r['question_id'], r['condition']):r for r in plan['records']}
    answers = {}
    for row in read_rows(args.run/'deliveries.jsonl'):
        key = row['question_id'], row['condition']
        record = expected.get(key)
        if (key in answers or record is None or row.get('run_config_sha256') != checksum
                or row.get('prompt') != plan['prompts'][key[1]] or row.get('interventions') != []
                or row.get('strength') != 0.
                or any(row.get(k) != record[k] for k in ('audio_sha256', 'audio_file', 'reference_answer'))):
            raise ValueError(f'Saved answer differs from frozen condition/audio: {key}')
        answers[key] = row
    eligible = [q for q in plan['question_order'] if all((q,c) in answers for c in plan['conditions'])]
    if args.question_id:
        if args.question_id not in plan['bank']: parser.error('Question ID absent from this run')
        selected = [args.question_id]
    elif args.longest:
        selected = sorted(eligible, key=lambda q: answers[q,'angry']['response_tokens'], reverse=True)[:args.count]
        print('Longest angry outputs; selected by length, not representative.')
    else:
        selected = random.Random(args.seed).sample(eligible, min(args.count, len(eligible)))
        print(f'Random matched sample, seed={args.seed}; no outcome or length filtering.')
    for q in selected:
        item = plan['bank'][q]
        print('\n'+'='*72+'\n'+q+'\nORIGINAL: '+item['original_question']+'\n'+
            '\n'.join(item['original_options'])+'\nGOLD: '+item['original_target']+'\nSPOKEN:\n'+item['question'])
        for c in plan['conditions']:
            row = answers.get((q,c))
            if row is None: print('\n'+c+': pending')
            else: print(f'\n{c}-bare | tokens={row["response_tokens"]}\nAUDIO: {row["audio_file"]}\n{row["response"]}')
    if not selected: print('No complete neutral/angry question pairs yet.')


if __name__ == '__main__': main()
