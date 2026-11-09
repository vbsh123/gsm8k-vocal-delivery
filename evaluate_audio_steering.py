"""Compare extracted vector methods on the same audio recordings.

Each method runs separately on one GPU. Seed42 baselines are shared between
methods; directions never share a generation. The full460 cohort is exploratory.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from compare_cloud_accuracy import PROMPT
from scoring import score_response
from steer import read_rows,summarize


CONDITIONS=[dict(name='baseline',interventions=[]),
            dict(name='L12-acoustic-a1',interventions=[dict(layer=12,group='acoustic',strength=1)]),
            dict(name='L16-answer-a1.25',interventions=[dict(layer=16,group='answer',strength=1.25)])]


def report(root,methods):
    for method in methods:
        folder=root/method
        if not (folder/'run_config.json').exists():
            print(method+': not started',flush=True)
            continue
        print('\nDirection method:',method,flush=True)
        config=json.loads((folder/'run_config.json').read_text())
        summarize(read_rows(folder/'results.jsonl'),config['settings']['conditions'])
    print('Same second-set recordings across methods; layer/strength choices were explored previously.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vectors-root',type=Path,default=Path('outputs/jenny-svd-probe'))
    p.add_argument('--source-run',type=Path,default=Path('outputs/steer-jenny-angry-complete'))
    p.add_argument('--output',type=Path,default=Path('outputs/steer-jenny-svd-probe-second460'))
    p.add_argument('--methods',nargs='+',choices=['mean','svd1','svd_mean','probe'],default=['svd_mean','probe'])
    p.add_argument('--baseline-source',type=Path,default=Path('outputs/steer-jenny-L16-full460-seed42'))
    p.add_argument('--num-examples',type=int,default=460)
    p.add_argument('--include-l11',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    if args.summary_only:
        report(args.output,args.methods)
        return
    conditions = list(CONDITIONS)
    if args.include_l11:
        conditions.insert(1, dict(name='L11-answer-a1', interventions=[
            dict(layer=11, group='answer', strength=1)]))
    source=json.loads((args.source_run/'run_config.json').read_text())
    records=source['recordings'][:args.num_examples]
    if not records or args.num_examples<1 or len(records)!=args.num_examples:
        p.error('Requested number of source recordings unavailable')
    ids={r['question_id'] for r in records}
    commands=[]
    for method in args.methods:
        vector=args.vectors_root/method/'style-vectors.pt'
        if not vector.is_file():
            p.error(f'Missing fitted vector: {vector}; finish collection/fitting first')
        excluded={r['question_id'] for r in read_rows(vector.parent/'extraction-questions.jsonl')}
        if ids & excluded:
            p.error('Extraction and evaluation questions overlap')
        folder=args.output/method
        folder.mkdir(parents=True,exist_ok=True)
        plan=dict(recordings=records,conditions=conditions,method=method,seed=42,
                  selection='Same original full460 second-set cohort; no correctness or classifier filter')
        path=folder/'selected-plan.json'
        if path.exists() and json.loads(path.read_text())!=plan:
            p.error('Saved evaluation cohort changed; use another output')
        path.write_text(json.dumps(plan,indent=2)+'\n')
        stat = vector.stat()
        settings = dict(previous=str(args.source_run.resolve()), vectors=str(vector.resolve()),
                        vector_bytes=stat.st_size, vector_mtime_ns=stat.st_mtime_ns,
                        seed=42, conditions=conditions, style='angry', mode='fixed',
                        selection='outcome-selected diagnostic',
                        seed_policy='constant_before_every_generation',
                        prepared_plan=str(path.resolve()),
                        prepared_plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        config = dict(settings=settings, recordings=records)
        config_path = folder/'run_config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            p.error('Saved run changed; use another output')
        config_path.write_text(json.dumps(config, indent=2)+'\n')
        command=[sys.executable,'-u',str(Path(__file__).with_name('steer_audio.py')),
                 '--config',str(config_path),'--output',str(folder)]
        subprocess.run(command+['--prepare-only'],check=True)
        commands.append((method,folder,command))
    if args.prepare_only:
        return
    shared={}
    expected={r['clip_id']:r for r in records}
    for row in read_rows(args.baseline_source/'results.jsonl'):
        if row['condition']!='baseline' or row['clip_id'] not in expected:
            continue
        if (row.get('seed')!=42 or row.get('prompt')!=PROMPT or row.get('max_new_tokens')!=512
                or row.get('decoder')!='fast_dllm' or row.get('steered_forwards')!=0
                or row.get('interventions')!=[]
                or str(row['reference_answer'])!=str(expected[row['clip_id']]['answer'])):
            continue
        shared[row['clip_id']]=row
    print(f'Reusing {len(shared)} matching seed42 baselines.',flush=True)
    for method,folder,command in commands:
        path=folder/'results.jsonl'
        saved={(r['clip_id'],r['condition']):r for r in read_rows(path)}
        with path.open('a') as stream:
            for cid,row in shared.items():
                key=(cid,'baseline')
                if key in saved:
                    if saved[key]!=row:
                        p.error(f'Conflicting same-seed baseline: {cid}')
                    continue
                stream.write(json.dumps(row)+'\n')
        print('\nEvaluating direction method:',method,flush=True)
        subprocess.run(command,check=True)
        for row in read_rows(path):
            if row['condition']!='baseline':
                continue
            if (row['seed']!=42 or row['prompt']!=PROMPT or row['max_new_tokens']!=512
                    or row['decoder']!='fast_dllm' or row['steered_forwards']!=0):
                p.error('Baseline recipe changed between methods')
            score=score_response(row['response'],row['reference_answer'])
            if row['correct']!=score['correct']:
                p.error('Baseline scoring differs')
            shared[row['clip_id']]=row
    report(args.output,args.methods)


if __name__=='__main__':
    main()
