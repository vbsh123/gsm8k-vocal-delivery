"""Inspect matched saved arithmetic answers; no generation, scoring or file changes."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from steer import read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('outputs/diffa-bbh-arithmetic50'))
    parser.add_argument('--count', type=int, default=3, help='Number of questions, each shown in every selected condition.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--conditions', nargs='+', help='Show only these condition names.')
    parser.add_argument('--question-id', help='Inspect a source question ID, including pending answers.')
    args = parser.parse_args()
    if args.count < 1: parser.error('--count must be positive')
    config = args.run/'run_config.json'
    if not config.exists():
        parser.exit(message='Arithmetic evaluation is not prepared yet; audio synthesis may still be running.\n')
    plan = json.loads(config.read_text())
    if plan.get('name') != 'multistep_arithmetic_two':
        parser.error('Expected a BBH arithmetic run configuration')
    conditions = args.conditions or plan['conditions']
    if len(conditions) != len(set(conditions)) or set(conditions)-set(plan['conditions']):
        parser.error('Select unique conditions from: '+', '.join(plan['conditions']))
    expected = {(r['question_id'], r['condition']):r for r in plan['records']}
    checksum = hashlib.sha256(config.read_bytes()).hexdigest()
    answers = {}
    for row in read_rows(args.run/'deliveries.jsonl'):
        key = row['question_id'], row['condition']
        record = expected.get(key)
        if (key in answers or record is None or row.get('run_config_sha256') != checksum
                or row.get('prompt') != plan['prompts'][key[1]]
                or any(row.get(k) != record[k] for k in ('audio_sha256', 'audio_file', 'reference_answer'))):
            raise ValueError(f'Saved answer differs from frozen condition/audio: {key}')
        answers[key] = row
    if args.question_id:
        if args.question_id not in plan['bank']: parser.error('Question ID absent from this run')
        selected = [args.question_id]
    else:
        eligible = [q for q in plan['question_order'] if all((q,c) in answers for c in conditions)]
        selected = random.Random(args.seed).sample(eligible, min(args.count, len(eligible)))
        print(f'Random matched sample, seed={args.seed}; {len(selected)} questions × {len(conditions)} conditions. '
              'No accuracy or response-length filtering.')
        if len(selected) < args.count:
            print(f'Only {len(eligible)} questions completed in all selected conditions so far.')
    for q in selected:
        item = plan['bank'][q]
        print('\n'+'='*72+'\n'+q+'\nEXPRESSION: '+item['original_input']+
              '\nSPOKEN QUESTION: '+item['question']+'\nGOLD: '+item['original_target'])
        for c in conditions:
            row = answers.get((q,c))
            if row is None:
                print('\n'+c+': pending')
            else:
                label = c.replace('-reasoning', '-supply-reasoning')
                if c in ('neutral', 'angry'): label += '-bare'
                print(f'\n{label} | tokens={row["response_tokens"]}\nAUDIO: {row["audio_file"]}\n{row["response"]}')


if __name__ == '__main__':
    main()
