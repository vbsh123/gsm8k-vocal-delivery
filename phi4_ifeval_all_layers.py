"""Three queued Phi workers: all32 audio and answer interventions at -1 on IFEval."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid

import phi4_sweep as sweep
import phi4_task_steering as task
from phi4_queued_workers import gpu_queue,run_workers
from kimi_bbh_layer_diagnostic import atomic_save,digest,freeze

OUTPUT=Path('outputs/phi4-ifeval-all-layers-minus1')


def jobs():
    return [(layer,group) for layer in range(32) for group in ('audio','answer')]


def assigned(items,worker_id):return list(items)[worker_id::3]


def cache_root(output):return output/'collection-cache/all-layers/ifeval'


def capture_config(data):
    recipe=dict(task.gsm.MODELS['phi4'],dtype='bfloat16',do_sample=False,num_beams=1,use_cache=True,repetition_penalty=1.,text_only_output=True)
    return dict(protocol='phi4-ifeval-all32-collection-v1',task='ifeval',collection=data['collection'],recipe=recipe,
        prompt=data['prompt'],seed=42,max_new_tokens=data['max_new_tokens'],groups=['answer','audio'],
        collector_sha256=digest(__file__),capture_sha256=digest(task.native_hooks.__file__),
        note='Equal questions; answer means over all native last-prediction forwards including prefill, audio means over prefill audio positions.')


def validate_pair(saved,pair,config_hash):
    import torch
    if saved['run_config_sha256']!=config_hash or saved['question_id']!=pair['question_id'] or saved['voice']!=pair['voice'] or saved['audio_sha256']!={s:pair[s]['audio_sha256'] for s in ('neutral','angry')}:
        raise ValueError('IFEval collection pair provenance differs.')
    for trace in saved['traces'].values():
        if trace['means'].shape!=task.native_hooks.SHAPE or not torch.isfinite(trace['means']).all() or trace['hook_forwards']<=0:raise ValueError('Invalid all-layer capture.')


def merge(output):
    import torch
    cache=cache_root(output);config=json.loads((cache/'run_config.json').read_text());config_hash=digest(cache/'run_config.json')
    differences=[]
    for i,pair in enumerate(config['collection']):
        saved=torch.load(cache/'pairs'/f'{i:04d}.pt',map_location='cpu',weights_only=True);validate_pair(saved,pair,config_hash)
        differences.append(saved['traces']['angry']['means']-saved['traces']['neutral']['means'])
    state=dict(protocol=config['protocol'],run_config_sha256=config_hash,recipe=config['recipe'],groups=['answer','audio'],
        collection_question_ids=[p['question_id'] for p in config['collection']],angry_minus_neutral=torch.stack(differences).mean(0))
    path=cache/'style-vectors.pt'
    if path.exists():
        previous=torch.load(path,map_location='cpu',weights_only=True)
        if previous['run_config_sha256']!=config_hash or not torch.equal(previous['angry_minus_neutral'],state['angry_minus_neutral']):raise ValueError('All-layer vector changed.')
    else:atomic_save(path,state)
    print(f'All{len(config["collection"])} collection pairs merged; releasing three evaluation workers for64 disjoint settings.',flush=True)


def prepare_sources(args):
    path=args.data_root/'ifeval/data_config.json'
    if not path.exists():task.prepare_data('ifeval',args.data_root)
    data=json.loads(path.read_text())
    if len(data['collection'])!=254 or len(data['test'])!=254 or data['max_new_tokens']!=1024:raise ValueError('Expected original254/254 IFEval split and1024-token budget.')
    cache=cache_root(args.output);cache.mkdir(parents=True,exist_ok=True);(cache/'pairs').mkdir(exist_ok=True)
    freeze(cache/'run_config.json',capture_config(data))
    for layer,group in jobs():
        out=args.output/'ifeval'/f'L{layer}-{group}-a-1';out.mkdir(parents=True,exist_ok=True)
        plan=task.make_plan(data,path,layer,group,-1.);freeze(out/'run_config.json',plan)
        sweep.reuse_baselines(plan,out,args.output)
        tasks,done=task.completed(plan,out)
        missing=[k for k in tasks if k[2] in ('neutral','angry') and k not in done]
        if missing:raise ValueError(f'Missing{len(missing)} existing IFEval baseline answers; no baseline generation scheduled.')
    return data


def worker(args):
    import torch
    from evaluate_bbh_candidates import concurrent_budget_now
    data_path=args.data_root/'ifeval/data_config.json';data=json.loads(data_path.read_text())
    cache=cache_root(args.output);config_hash=digest(cache/'run_config.json')
    indices=assigned(range(len(data['collection'])),args.worker_id);settings=assigned(jobs(),args.worker_id)
    print(f'Worker{args.worker_id}: {len(indices)} collection questions; evaluation settings {settings}',flush=True)
    concurrent_budget_now(15.)
    native=task.Native(task.make_plan(data,data_path,0,'answer',-1.))
    (args.output/f'runtime-worker{args.worker_id}.json').write_text(json.dumps(dict(torch=torch.__version__,
        transformers=task.engine.version('transformers'),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        torch_cap_gib=15.,collector_sha256=digest(__file__),task_evaluator_sha256=digest(task.__file__)),indent=2)+'\n')
    for i in indices:
        pair=data['collection'][i];path=cache/'pairs'/f'{i:04d}.pt'
        if path.exists():validate_pair(torch.load(path,map_location='cpu',weights_only=True),pair,config_hash);continue
        print(f'Collect {i+1}/254 {pair["question_id"]}: all32 answer+audio groups.',flush=True)
        traces={s:task.native_hooks.generate(native.native,pair[s],data['prompt'],collect=True,max_new_tokens=1024) for s in ('neutral','angry')}
        saved=dict(run_config_sha256=config_hash,question_id=pair['question_id'],voice=pair['voice'],
            audio_sha256={s:pair[s]['audio_sha256'] for s in traces},traces=traces)
        validate_pair(saved,pair,config_hash);atomic_save(path,saved)
    marker=args.output/'barriers'/args.token;marker.mkdir(parents=True,exist_ok=True)
    (marker/f'worker{args.worker_id}.ready').write_text(str(os.getpid())+'\n')
    while not (marker/'release').exists():
        try:os.kill(args.parent_pid,0)
        except ProcessLookupError:raise RuntimeError('Parent launcher exited before merge; collection caches retained.')
        time.sleep(1)
    for layer,group in settings:
        out=args.output/'ifeval'/f'L{layer}-{group}-a-1';plan=json.loads((out/'run_config.json').read_text())
        with sweep.job_lock(out):
            print(f'EVALUATE L{layer} {group} x-1; original1024-token budget.',flush=True)
            task.evaluate(plan,out,args.output/'collection-cache',native=native)
        # Scoring is CPU-only; native model remains loaded across settings.
        from types import SimpleNamespace
        sweep.score_in_cpu_environment(SimpleNamespace(output=args.output,data_root=args.data_root,group=group), 'ifeval',layer,-1.)


def summary(args):
    for layer,group in jobs():task.report(args.output/'ifeval'/f'L{layer}-{group}-a-1')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('run','summary'),default='run')
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--data-root',type=Path,default=Path('outputs/phi4-next-tasks'))
    p.add_argument('--worker-id',type=int,choices=range(3));p.add_argument('--token');p.add_argument('--parent-pid',type=int)
    args=p.parse_args()
    if args.mode=='summary':summary(args);return
    if args.worker_id is not None:worker(args);return
    args.output.mkdir(parents=True,exist_ok=True)
    with sweep.job_lock(args.output/'launcher'):
        prepare_sources(args)
        freeze(args.output/'assignment.json',dict(protocol='phi4-ifeval-all32-audio-answer-minus1-v1',
            collection_by_worker=[assigned(range(254),i) for i in range(3)],jobs_by_worker=[assigned(jobs(),i) for i in range(3)],
            strength=-1.,workers=3,max_new_tokens=1024,note='Exploratory all-layer sweep; distinct jobs per worker; no outcome selection.'))
        with gpu_queue():
            token=uuid.uuid4().hex;marker=args.output/'barriers'/token;marker.mkdir(parents=True)
            commands=[['.venv-bbh-phi4/bin/python','-u',str(Path(__file__).resolve()),'--worker-id',str(i),
                '--output',str(args.output),'--data-root',str(args.data_root),'--token',token,'--parent-pid',str(os.getpid())] for i in range(3)]
            run_workers(commands,args.output,ready_paths=[marker/f'worker{i}.ready' for i in range(3)],release=marker/'release',on_ready=lambda:merge(args.output))

if __name__=='__main__':main()
