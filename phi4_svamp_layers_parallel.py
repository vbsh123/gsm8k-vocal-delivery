"""Three persistent Phi workers, disjoint SVAMP layers, shared multi-layer captures."""
import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import subprocess
import time

import phi4_sweep as sweep
import phi4_task_steering as task
from kimi_bbh_layer_diagnostic import atomic_save, digest, freeze


def partitions(layers, workers):
    return [layers[i::workers] for i in range(workers)]


def collection_config(plan):
    common={k:plan[k] for k in ('task','collection','recipe','prompt','seed','max_new_tokens','layer','group','vector_definition')}
    return dict(common,protocol='phi4-sweep-collection-v1')


def collect_together(plans, root, native):
    """Generate each missing delivery once, observing every assigned layer together."""
    import torch
    with ExitStack() as stack:
        caches={}
        for plan in plans:
            cache=root/plan['task']/f"L{plan['layer']}-{plan['group']}"
            cache.mkdir(parents=True,exist_ok=True)
            lock=stack.enter_context((cache/'collection.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            common=collection_config(plan);freeze(cache/'run_config.json',common)
            (cache/'pairs').mkdir(exist_ok=True)
            caches[plan['layer']]=(common,cache,digest(cache/'run_config.json'))
        first=plans[0]
        for i,pair in enumerate(first['collection']):
            missing=[]
            for layer,(common,cache,config_hash) in caches.items():
                path=cache/'pairs'/f'{i:04d}.pt'
                if path.exists():
                    task.engine.validate_pair(torch.load(path,map_location='cpu',weights_only=True),pair,config_hash,common['protocol'])
                else:missing.append(layer)
            if not missing:continue
            print(f'Collect {i+1}/{len(first["collection"])} {pair["question_id"]}; simultaneous layers {missing}',flush=True)
            traces={style:task.native_hooks.generate(native.native,pair[style],first['prompt'],collect=True,
                capture_layers=missing,max_new_tokens=first['max_new_tokens']) for style in ('neutral','angry')}
            for index,layer in enumerate(missing):
                common,cache,config_hash=caches[layer]
                selected={style:dict((k,v) for k,v in trace.items() if k!='means') for style,trace in traces.items()}
                for style in selected:
                    selected[style]['mean']=traces[style]['means'][index,task.native_hooks.GROUPS.index(first['group'])].clone()
                saved=dict(protocol=common['protocol'],run_config_sha256=config_hash,question_id=pair['question_id'],voice=pair['voice'],
                    audio_sha256={s:pair[s]['audio_sha256'] for s in selected},traces=selected,
                    answer_difference=selected['angry']['mean']-selected['neutral']['mean'])
                task.engine.validate_pair(saved,pair,config_hash,common['protocol']);atomic_save(cache/'pairs'/f'{i:04d}.pt',saved)
        # Finalize the usual vector files using cached pairs; no additional generations.
        for common,cache,config_hash in caches.values():
            task.engine.collect(common,cache,native,config_hash)


def worker(args):
    layers=partitions(args.layers,args.workers)[args.worker_id]
    print(f'Worker {args.worker_id}: layers {layers}, strengths {args.strengths}; one persistent Phi model.',flush=True)
    path=args.data_root/'svamp/data_config.json'
    data,path=sweep.subset_data(args,json.loads(path.read_text()),path)
    plans=[task.make_plan(data,path,layer,'answer',args.strengths[0]) for layer in layers]
    with sweep.job_lock(args.output/f'nightly-worker{args.worker_id}'):
        from evaluate_bbh_candidates import concurrent_budget_now
        concurrent_budget_now(18.)
        native=task.Native(plans[0])
        collect_together(plans,args.output/'collection-cache',native)
        print(f'Worker {args.worker_id}: collection complete; evaluating assigned layers.',flush=True)
        args.mode='run';args.layers=layers;args.datasets=['svamp'];args.group='answer'
        sweep.run(args,native=native)


def wait_for_gpu(stack):
    lock=stack.enter_context(Path('outputs/jenny-task-vectors.lock').open('a'))
    print('Waiting for shared GPU lock; all three workers start together afterward.',flush=True)
    fcntl.flock(lock,fcntl.LOCK_EX)
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        print('Waiting for active GPU work to finish.',flush=True)
        time.sleep(30)


def launch(args):
    args.output.mkdir(parents=True,exist_ok=True)
    with sweep.job_lock(args.output/'nightly-launcher'),ExitStack() as stack:
        prepare=['bash','run_phi4_sweep.sh','--mode','prepare','--datasets','svamp',
            '--data-root',str(args.data_root),'--output',str(args.output),*sweep.cohort_options(args)]
        subprocess.run(prepare,check=True)
        assignments=partitions(args.layers,args.workers)
        freeze(args.output/'nightly-assignment.json',dict(protocol='phi4-svamp-parallel-layers-v1',
            layers_by_worker=assignments,strengths=args.strengths,collection_size=args.collection_size,test_size=args.test_size,
            note='Exploratory sweep on previously inspected test questions; no GPU queue.'))
        if getattr(args,'wait_for_gpu',False):wait_for_gpu(stack)
        children=[]
        try:
            for i,layers in enumerate(assignments):
                log_path=args.output/f'nightly-worker{i}.log'
                log=stack.enter_context(log_path.open('a'))
                command=['.venv-bbh-phi4/bin/python','-u',str(Path(__file__).resolve()),'--worker-id',str(i),
                    '--workers',str(args.workers),'--layers',*[str(x) for x in args.layers],
                    '--strengths',*[str(x) for x in args.strengths],'--output',str(args.output),
                    '--data-root',str(args.data_root),*sweep.cohort_options(args)]
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                children.append(process)
                print(f'STARTED worker{i} PID{process.pid}: layers {layers}; log {log_path}',flush=True)
            failures=[]
            for i,child in enumerate(children):
                code=child.wait()
                if code:failures.append((i,code))
            if failures:raise RuntimeError(f'Failed workers: {failures}. Saved outputs retained; inspect worker logs.')
        finally:
            for child in children:
                if child.poll() is None:child.terminate()
            for child in children:child.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers',type=int,nargs='+',default=[i for i in range(32) if i not in (17,28,29,30)])
    parser.add_argument('--strengths',type=float,nargs='+',default=[-1.,1.])
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--worker-id',type=int)
    parser.add_argument('--wait-for-gpu',action='store_true',help='Parent waits for shared GPU lock and no active GPU processes, then starts all workers together.')
    parser.add_argument('--collection-size',type=int,default=300)
    parser.add_argument('--test-size',type=int,default=300)
    parser.add_argument('--data-root',type=Path,default=Path('outputs/phi4-next-tasks'))
    parser.add_argument('--output',type=Path,default=Path('outputs/phi4-sweeps/svamp300-300'))
    args=parser.parse_args()
    import math
    args.layers=list(dict.fromkeys(args.layers));args.strengths=list(dict.fromkeys(args.strengths))
    if any(i not in range(32) for i in args.layers):parser.error('Layers must be 0..31.')
    if not 1<=args.workers<=len(args.layers):parser.error('Workers must be between 1 and the number of layers.')
    if args.worker_id is not None and not 0<=args.worker_id<args.workers:parser.error('Invalid worker ID.')
    if any(not math.isfinite(s) or s==0 for s in args.strengths):parser.error('Strengths must be finite and nonzero.')
    if args.collection_size<=0 or args.test_size<=0:parser.error('Cohort sizes must be positive.')
    if args.worker_id is None:launch(args)
    else:worker(args)


if __name__=='__main__':main()
