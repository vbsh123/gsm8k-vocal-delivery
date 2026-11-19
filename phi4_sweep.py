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
    paths+=list((Path('outputs/phi4-sweeps')/dataset).glob('*/results.jsonl'))
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
                    or row.get('seed')!=42 or row.get('max_new_tokens')!=plan['max_new_tokens'] or row.get('reference_answer')!=expected['reference_answer']
                    or row.get('model')!=plan['recipe']['model'] or row.get('revision')!=plan['recipe']['revision']
                    or row.get('strength',0)!=0 or row.get('interventions')):continue
                saved=dict(expected,response=row['response'],response_tokens=row['response_tokens'],prompt=plan['prompt'],seed=42,max_new_tokens=plan['max_new_tokens'],
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


def group_options(args):
    return ['--group',getattr(args,'group','answer')]


def cohort_options(args):
    result=[]
    for name in ('collection_size','test_size'):
        value=getattr(args,name,None)
        if value is not None:result.extend(['--'+name.replace('_','-'),str(value)])
    return result


def subset_data(args,data,path):
    collection_size=getattr(args,'collection_size',None);test_size=getattr(args,'test_size',None)
    if collection_size is None and test_size is None:return data,path
    c=collection_size if collection_size is not None else len(data['collection'])
    t=test_size if test_size is not None else len(data['test'])
    if not 0<c<=len(data['collection']) or not 0<t<=len(data['test']):raise ValueError('Requested cohort exceeds the existing frozen halves.')
    collection=data['collection'][:c];test=data['test'][:t]
    train_ids={p['question_id'] for p in collection};test_ids={p['question_id'] for p in test}
    if len(train_ids)!=c or len(test_ids)!=t or train_ids & test_ids:raise ValueError('Duplicate or overlapping collection/test questions.')
    selected=dict(data,collection=collection,test=test,source_data_config=str(path.resolve()),source_data_config_sha256=digest(path),
        cohort_selection=f'First{c} collection and first{t} test questions from the existing separately frozen halves; no outcome filtering. Test cohort may have been inspected previously.')
    target=args.output/'cohorts'/data['task']/'data_config.json';target.parent.mkdir(parents=True,exist_ok=True);freeze(target,selected)
    print(f'{data["task"]}: selected {c} collection /{t} disjoint test questions from existing halves.',flush=True)
    return selected,target


def run(args):
    group=getattr(args,'group','answer')
    for dataset in args.datasets:
        path=args.data_root/dataset/'data_config.json'
        if args.mode=='prepare' or not path.exists():task.prepare_data(dataset,args.data_root)
        data,path=subset_data(args,json.loads(path.read_text()),path)
        if args.mode=='prepare':continue
        for layer in args.layers:
            for strength in args.strengths:
                output=args.output/dataset/f'L{layer}-{group}-a{strength:g}'
                if args.mode=='summary':task.report(output);continue
                with job_lock(output):
                    plan=task.make_plan(data,path,layer,group,strength)
                    freeze(output/'run_config.json',plan);reuse_baselines(plan,output,args.output)
                    print(f'STAGE: {dataset} L{layer} {group} x{strength:g}; {len(data["collection"])} collection /{len(data["test"])} test.',flush=True)
                    task.evaluate(plan,output,args.output/'collection-cache')
                    score_in_cpu_environment(args,dataset,layer,strength)


def score_in_cpu_environment(args,dataset,layer,strength):
    python=os.environ.get('DIFFA_PYTHON','.venv-diffa/bin/python')
    subprocess.run([python,'-u',str(Path(__file__).resolve()),'--mode','summary',
        '--datasets',dataset,'--layers',str(layer),'--strengths',str(strength),
        '--data-root',str(args.data_root),'--output',str(args.output),*cohort_options(args),*group_options(args)],check=True)


def parallel(args):
    """Prepare audio once, reserve the GPU once, then launch one worker per layer."""
    with job_lock(args.output/'parallel-launcher'), ExitStack() as stack:
        cpu=os.environ.get('DIFFA_PYTHON','.venv-diffa/bin/python')
        print('STAGE: Prepare/reuse selected datasets and audio once on CPU.',flush=True)
        subprocess.run([cpu,'-u',str(Path(__file__).resolve()),'--mode','prepare',
            '--datasets',*args.datasets,'--data-root',str(args.data_root),'--output',str(args.output),*cohort_options(args),*group_options(args)],check=True)
        if not args.no_wait:
            gpu=stack.enter_context(Path('outputs/jenny-task-vectors.lock').open('a'))
            print('Waiting for shared GPU lock; two layer workers start together afterward.',flush=True)
            fcntl.flock(gpu,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
        workers=[]
        try:
            for layer in args.layers:
                log_path=args.output/f'worker-L{layer}.log';log=stack.enter_context(log_path.open('a'))
                command=['.venv-bbh-phi4/bin/python','-u',str(Path(__file__).resolve()),'--mode','run','--no-wait',
                    '--datasets',*args.datasets,'--layers',str(layer),'--strengths',*[str(s) for s in args.strengths],
                    '--data-root',str(args.data_root),'--output',str(args.output),*cohort_options(args),*group_options(args)]
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                workers.append((layer,process));print(f'STARTED L{layer}, PID{process.pid}; log: {log_path}',flush=True)
            failed=[]
            for layer,process in workers:
                code=process.wait();print(f'L{layer} worker finished: exit{code}',flush=True)
                if code:failed.append((layer,code))
            if failed:raise SystemExit(f'Failed layer workers: {failed}; saved results retained. Check their logs and resume.')
        except BaseException:
            # Only stop children belonging to this launcher if it is interrupted.
            for _,process in workers:
                if process.poll() is None:process.terminate()
            for _,process in workers:process.wait()
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','run','summary','parallel'),default='run')
    p.add_argument('--datasets',nargs='+',choices=tuple(task.TASKS),default=list(task.TASKS))
    p.add_argument('--layers',type=int,nargs='+',default=[2])
    p.add_argument('--group',choices=('answer','audio'),default='answer')
    p.add_argument('--strengths',type=float,nargs='+',default=[-1.,-1.5])
    p.add_argument('--data-root',type=Path,default=Path('outputs/phi4-next-tasks'))
    p.add_argument('--output',type=Path,default=Path('outputs/phi4-sweeps'))
    p.add_argument('--collection-size',type=int,help='Use a prefix of the frozen collection half; requires a separate --output.')
    p.add_argument('--test-size',type=int,help='Use a prefix of the frozen test half; requires a separate --output.')
    p.add_argument('--no-wait',action='store_true',help='Skip shared GPU queue; use only when enough GPU memory is available.')
    args=p.parse_args()
    if (args.collection_size is not None or args.test_size is not None) and args.output==Path('outputs/phi4-sweeps'):
        p.error('Use a separate --output for a custom-size cohort to preserve existing runs.')
    if any(n is not None and n<=0 for n in (args.collection_size,args.test_size)):p.error('Cohort sizes must be positive.')
    if any(layer not in range(32) for layer in args.layers):p.error('Layers must be0..31.')
    if any(not math.isfinite(s) or s==0 for s in args.strengths):p.error('Supply finite nonzero strengths; baseline0 is automatic.')
    args.datasets=list(dict.fromkeys(args.datasets));args.layers=list(dict.fromkeys(args.layers));args.strengths=list(dict.fromkeys(args.strengths))
    if args.mode=='parallel':
        if len(args.layers)!=2:p.error('Parallel mode needs exactly two distinct layers.')
        args.output.mkdir(parents=True,exist_ok=True);parallel(args);return
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
