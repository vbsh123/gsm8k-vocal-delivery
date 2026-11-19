"""Sequential own-task Phi sweeps with chosen datasets, layers and signed strengths."""
import argparse
from contextlib import ExitStack,contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

import phi4_task_steering as task
from kimi_bbh_layer_diagnostic import digest,freeze
from steer import read_rows


def baseline_sources(dataset,root):
    paths=list((root/dataset).glob('*/results.jsonl'))
    paths+=list((Path('outputs/phi4-task-steering')/dataset).glob('*/results.jsonl'))
    if dataset=='gsm8k':
        paths+=list(Path('outputs/phi4-gsm400-all-layers-test869-minus1').glob('references*.jsonl'))
        paths+=list(Path('outputs/gsm8k-candidates/phi4/original').glob('results.jsonl'))
        paths+=list(Path('outputs').glob('phi4-gsm-reverse*/results.jsonl'))
    elif dataset=='multiarith':paths+=list(Path('outputs').glob('phi4-multiarith-reverse*/results.jsonl'))
    return sorted(set(paths))


def reuse_baselines(plan,output,root):
    targets,done=task.completed(plan,output,repair=True);count=0
    with (output/'results.jsonl').open('a') as stream:
        for source in baseline_sources(plan['task'],root):
            if source==output/'results.jsonl':continue
            for row in read_rows(source):
                condition=row.get('condition',row.get('style'));key=row['question_id'],row['voice'],condition
                if condition not in ('neutral','angry') or key not in targets or key in done:continue
                expected=targets[key]
                if (row.get('audio_sha256')!=expected['audio_sha256'] or row.get('prompt')!=plan['prompt']
                    or row.get('seed')!=42 or row.get('max_new_tokens')!=512 or row.get('reference_answer')!=expected['reference_answer']
                    or row.get('model')!=plan['recipe']['model'] or row.get('revision')!=plan['recipe']['revision']
                    or row.get('strength',0)!=0 or row.get('interventions')):continue
                saved=dict(expected,response=row['response'],response_tokens=row['response_tokens'],prompt=plan['prompt'],seed=42,max_new_tokens=512,
                    model=plan['recipe']['model'],revision=plan['recipe']['revision'],interventions=[],
                    run_config_sha256=digest(output/'run_config.json'),reused_from=str(source))
                stream.write(json.dumps(saved)+'\n');done[key]=saved;count+=1
    print(f'Reused {count} compatible unsteered answers.',flush=True)


@contextmanager
def job_lock(output):
    """Different settings may run concurrently; duplicate writers cannot."""
    output.mkdir(parents=True,exist_ok=True)
    with (output/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(f'Already running: {output}') from None
        yield


def run(args):
    for dataset in args.datasets:
        path=args.data_root/dataset/'data_config.json'
        if args.mode=='prepare' or not path.exists():task.prepare_data(dataset,args.data_root)
        if args.mode=='prepare':continue
        data=json.loads(path.read_text())
        for layer in args.layers:
            for strength in args.strengths:
                output=args.output/dataset/f'L{layer}-answer-a{strength:g}'
                if args.mode=='summary':task.report(output);continue
                with job_lock(output):
                    plan=task.make_plan(data,path,layer,'answer',strength)
                    freeze(output/'run_config.json',plan);reuse_baselines(plan,output,args.output)
                    print(f'STAGE: {dataset} L{layer} answer x{strength:g}; {len(data["collection"])} collection /{len(data["test"])} test.',flush=True)
                    task.evaluate(plan,output,args.output/'collection-cache')
                    score_in_cpu_environment(args,dataset,layer,strength)


def score_in_cpu_environment(args,dataset,layer,strength):
    python=os.environ.get('DIFFA_PYTHON','.venv-diffa/bin/python')
    subprocess.run([python,'-u',str(Path(__file__).resolve()),'--mode','summary',
        '--datasets',dataset,'--layers',str(layer),'--strengths',str(strength),
        '--data-root',str(args.data_root),'--output',str(args.output)],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','run','summary'),default='run')
    p.add_argument('--datasets',nargs='+',choices=tuple(task.TASKS),default=list(task.TASKS))
    p.add_argument('--layers',type=int,nargs='+',default=[2])
    p.add_argument('--strengths',type=float,nargs='+',default=[-1.,-1.5])
    p.add_argument('--data-root',type=Path,default=Path('outputs/phi4-next-tasks'))
    p.add_argument('--output',type=Path,default=Path('outputs/phi4-sweeps'))
    p.add_argument('--no-wait',action='store_true',help='Skip shared GPU queue; use only when enough GPU memory is available.')
    args=p.parse_args()
    if any(layer not in range(32) for layer in args.layers):p.error('Layers must be0..31.')
    if any(not math.isfinite(s) or s==0 for s in args.strengths):p.error('Supply finite nonzero strengths; baseline0 is automatic.')
    args.datasets=list(dict.fromkeys(args.datasets));args.layers=list(dict.fromkeys(args.layers));args.strengths=list(dict.fromkeys(args.strengths))
    if args.mode!='run':run(args);return
    args.output.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        if not args.no_wait:
            gpu=stack.enter_context(Path('outputs/jenny-task-vectors.lock').open('w'))
            print('Waiting for shared GPU lock.',flush=True);fcntl.flock(gpu,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
        run(args)


if __name__=='__main__':main()
