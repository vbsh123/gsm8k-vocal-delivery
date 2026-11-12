"""Run existing Azure batches sequentially, resuming saved answers and predictions."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def run(command):
    print('Running: '+' '.join(map(str,command)),flush=True)
    subprocess.run(list(map(str,command)),check=True)


def environment_python(name):
    envs=json.loads(subprocess.check_output(['conda','env','list','--json'],text=True))['envs']
    prefix=next((Path(p) for p in envs if Path(p).name==name),None)
    if prefix is None or not (prefix/'bin/python').is_file():
        raise FileNotFoundError(f'Missing {name} Python environment; run its setup script first.')
    return str(prefix/'bin/python')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['diffa','qwen','kimi','classify','steering','report'])
    p.add_argument('--paths',nargs='+',type=Path,default=[Path('outputs/azure-jenny-next500'),Path('outputs/azure-jenny-remaining-test')])
    p.add_argument('--styles',nargs='+',default=['neutral','angry','happy','sad','terrified','shouting'])
    p.add_argument('--vectors',type=Path)
    p.add_argument('--steering-path',type=Path,default=Path('outputs/azure-jenny-remaining-test'))
    p.add_argument('--device',default='cuda:0',help='Classifier device; cpu leaves the GPU available.')
    p.add_argument('--results-file',default='results-matched-accuracy.jsonl',help='Used by report stage.')
    args=p.parse_args()
    for folder in ([args.steering_path] if args.stage=='steering' else args.paths):
        if not (folder/'plan.jsonl').exists():
            p.error(f'Missing generated recording plan: {folder}/plan.jsonl')
    commands=[]
    if args.stage=='diffa':
        commands=[[Path('.venv-diffa/bin/python'),'-u','compare_cloud_accuracy.py','--path',folder,
                   '--all-cloud','--styles',*args.styles] for folder in args.paths]
    elif args.stage in ('qwen','kimi'):
        env='qwen-omni' if args.stage=='qwen' else 'kimi-audio'
        model_python=environment_python(env)
        os.environ.pop('PYTHONHOME',None)
        os.environ.pop('PYTHONPATH',None)
        os.environ['PATH']=str(Path(model_python).parent)+os.pathsep+os.environ.get('PATH','')
        commands=[[model_python,'-u',
                   'evaluate_qwen_omni.py' if args.stage=='qwen' else 'evaluate_kimi_audio.py',
                   '--paths',*args.paths,'--styles',*args.styles,'--num-questions','0','--seed','42']]
    elif args.stage=='classify':
        commands=[['.venv-emotion/bin/python','-u','classify_cloud_smoke.py','--path',folder,
                   '--model',model,'--device',args.device] for folder in args.paths for model in ('emotion2vec','sensevoice')]
    elif args.stage=='steering':
        command=['.venv-diffa/bin/python','-u','steer_azure_holdout.py','--path',args.steering_path]
        if args.vectors:
            command.extend(['--vectors',args.vectors])
        commands=[command]
    if commands:
        if args.stage=='classify' and args.device=='cpu':
            for command in commands:
                run(command)
        else:
            Path('outputs').mkdir(exist_ok=True)
            with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
                print('Waiting for the shared GPU lock.',flush=True)
                fcntl.flock(lock,fcntl.LOCK_EX)
                while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                    print('Waiting for active GPU work to finish.',flush=True)
                    time.sleep(30)
                for command in commands:
                    run(command)
    if args.stage in ('diffa','qwen','kimi','classify','report'):
        results={'qwen':'results-qwen-omni.jsonl','kimi':'results-kimi-audio.jsonl'}.get(args.stage,args.results_file)
        report_python=['.venv-emotion/bin/python'] if args.stage=='classify' else ['.venv-diffa/bin/python']
        if args.stage in ('qwen','kimi'):
            report_python=[model_python]
        for folder in args.paths:
            print(f'\nBatch: {folder}',flush=True)
            run([*report_python,'analyze_jenny_deliveries.py','--paths',folder,'--styles',*args.styles,'--results-file',results,'--confidence-intervals'])
        print('\nCombined batches',flush=True)
        run([*report_python,'analyze_jenny_deliveries.py','--paths',*args.paths,'--styles',*args.styles,'--results-file',results,'--confidence-intervals'])


if __name__=='__main__':
    main()
