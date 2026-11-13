"""Read-only audit of saved Qwen/Kimi delivery answers; no model or GPU loading."""

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from compare_cloud_accuracy import PROMPT, read, summarize_deliveries
from scoring import score_response


def audit(paths, model):
    tasks, answers, labels, recipes = [], {}, {}, set()
    identities = {}
    for folder in paths:
        planned = {r['audio_path']: r for r in read(folder/'plan.jsonl') if r['provider']=='azure'}
        settings = json.loads((folder/f'{model}-settings.json').read_text())
        rows = read(folder/f'results-{model}.jsonl')
        seen = {}
        for row in rows:
            name = row['audio_path']
            if name not in planned:
                raise ValueError(f'{folder}: answer has no planned recording: {name}')
            task = planned[name]
            key = tuple(task[k] for k in ('question_id','voice','style'))
            if any(row[k]!=task[k] for k in ('question_id','voice','style')):
                raise ValueError(f'{folder}: answer identity differs from plan: {name}')
            gold = str(task['answer']).rsplit('####',1)[-1].strip().replace(',','')
            if Fraction(gold)!=Fraction(str(row['reference_answer']).replace(',','')):
                raise ValueError(f'{folder}: reference differs from plan: {name}')
            if row.get('seed')!=42 or row.get('prompt')!=PROMPT:
                raise ValueError(f'{folder}: unexpected seed or prompt: {name}')
            if any(row.get(k)!=v for k,v in settings.items()):
                raise ValueError(f'{folder}: answer recipe differs from saved settings: {name}')
            score = score_response(row['response'],gold)
            if any(row.get(k)!=score[k] for k in ('correct','predicted_answer','parse_failed')):
                raise ValueError(f'{folder}: saved score differs from fresh scoring: {name}')
            if name in seen:
                raise ValueError(f'{folder}: duplicate answer: {name}')
            if key in identities:
                raise ValueError(f'Recording counted in multiple folders: {key}')
            seen[name] = row
            identities[key] = folder
        recipes.add(json.dumps(settings,sort_keys=True))
        prefix = str(folder.resolve())+'/'
        cohort_tasks = [{**r,'audio_path':prefix+r['audio_path']} for r in planned.values()]
        cohort_answers = {prefix+name:row for name,row in seen.items()}
        tasks.extend(cohort_tasks);answers.update(cohort_answers)
        label_path = folder/'results-emotion2vec.jsonl'
        if label_path.exists():
            labels.update({prefix+r['audio_path']:r for r in read(label_path)})
        print(f'{folder}: {len(rows)} unique answers; recipe, references and scores verified.')
        print('  Deliveries:',dict(Counter(r['style'] for r in rows)))
        summarize_deliveries(cohort_tasks,cohort_answers,['neutral','angry','happy','sad','terrified','shouting'])
    if len(recipes)!=1:
        raise ValueError('Generation settings differ across cohorts; do not pool without reviewing the difference')
    print('\nAUDIT PASSED: same recipe across folders; no duplicate answers or score/reference mismatches.')
    summarize_deliveries(tasks,answers,['neutral','angry','happy','sad','terrified','shouting'])
    if labels:
        summarize_deliveries(tasks,answers,['neutral','angry','happy','sad'],labels)
    return len(answers)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',type=Path,nargs='+',required=True)
    p.add_argument('--model',choices=['qwen-omni','kimi-audio'],required=True)
    args = p.parse_args()
    audit(args.paths,args.model)


if __name__=='__main__':
    main()
