"""Print saved Object Counting answers side by side. CPU only; no scoring or generation."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from steer import read_rows


def load(folder):
    path=folder/'run_config.json'
    plan=json.loads(path.read_text())
    if plan.get('name')!='object_counting':
        raise ValueError(f'Not an Object Counting configuration: {path}')
    checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    expected={(r['question_id'],r['condition']):r for r in plan['records']}
    answers={}
    for row in read_rows(folder/'deliveries.jsonl'):
        key=row['question_id'],row['condition'];record=expected.get(key)
        if (key in answers or record is None or row.get('run_config_sha256')!=checksum
                or row.get('prompt')!=plan['prompts'][key[1]]
                or row.get('audio_sha256')!=record['audio_sha256']):
            raise ValueError(f'Saved answer differs from its frozen condition: {folder}/{key}')
        answers[key]=row
    return plan,checksum,answers


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('outputs/diffa-object-counting-prompts'))
    parser.add_argument('--extra',type=Path,default=Path('outputs/diffa-object-counting-angry-reasoning'))
    parser.add_argument('--supply-run',type=Path,default=Path('outputs/diffa-object-counting-supply-reasoning'))
    parser.add_argument('--count',type=int,default=3)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--longest',action='store_true',help='Inspect longest reasoning answers among the selected conditions instead of a random sample.')
    parser.add_argument('--question-id',help='Show a particular question, including pending conditions.')
    parser.add_argument('--conditions',nargs='+',help='Show only these conditions; random samples require all selected answers.')
    args=parser.parse_args()
    if args.count<1:parser.error('--count must be positive')
    plan,checksum,answers=load(args.run)
    conditions=list(plan['conditions'])
    for folder in (args.extra,args.supply_run):
        if not (folder/'run_config.json').exists():continue
        extra,_,other=load(folder)
        if extra.get('parent_config_sha256')!=checksum:
            raise ValueError(f'Extra reasoning run refers to a different original configuration: {folder}')
        extra_conditions=list(dict.fromkeys(r['condition'] for r in extra['records']))
        if set(extra_conditions)&set(conditions):raise ValueError('Duplicate condition names across runs.')
        answers.update(other)
        conditions.extend(extra_conditions)
    if args.conditions:
        unknown=set(args.conditions)-set(conditions)
        if unknown:parser.error('Conditions not prepared: '+', '.join(sorted(unknown)))
        if len(set(args.conditions))!=len(args.conditions):parser.error('Duplicate selected conditions')
        conditions=args.conditions
    required=conditions if args.conditions else plan['conditions']
    if args.question_id:
        if args.question_id not in plan['bank']:parser.error('Question ID absent from this run')
        selected=[args.question_id]
        print('Selected question ID; missing answers shown as pending.')
    else:
        eligible=[q for q in plan['question_order'] if all((q,c) in answers for c in required)]
        if args.longest:
            rank=next((c for c in ('neutral-reasoning','neutral-supply-reasoning') if c in conditions),conditions[0])
            selected=sorted(eligible,key=lambda q:(-answers[q,rank]['response_tokens'],q))[:args.count]
            print(f'Selected by longest {rank} output; diagnostic examples, not a representative sample.')
        else:
            selected=random.Random(args.seed).sample(eligible,min(args.count,len(eligible)))
            print(f'Random sample, seed={args.seed}; requires completed selected conditions; no accuracy filtering.')
    if not selected:
        print('No questions completed in the required conditions yet.')
    for q in selected:
        item=plan['bank'][q]
        print('\n'+'='*72+'\n'+q+'\nQUESTION: '+item['question']+'\nGOLD: '+item['original_target'])
        for c in conditions:
            row=answers.get((q,c))
            if row is None:
                print('\n'+c+': pending')
            else:
                print(f'\n{c} | tokens={row["response_tokens"]}\nAUDIO: {row["audio_file"]}\n{row["response"]}')


if __name__=='__main__':main()
