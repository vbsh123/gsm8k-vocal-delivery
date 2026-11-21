"""Queue BBH3/BBH5 own-vector L0 audio -1 + L30 answer -2, combined test only."""
import argparse
import json
from pathlib import Path

import phi4_gsm_combo as combo
import phi4_sweep as sweep
import phi4_task_steering as task
from phi4_queued_workers import gpu_queue,run_workers
from kimi_bbh_layer_diagnostic import freeze

OUTPUT=Path('outputs/phi4-bbh-combined')
DATASETS=('bbh3','bbh5')


def sources(name,data,data_path,output):
    result=[]
    for layer,group,strength in ((0,'audio',-1.),(30,'answer',-2.)):
        folder=output/name/'sources'/f'L{layer}-{group}';folder.mkdir(parents=True,exist_ok=True)
        plan=task.make_plan(data,data_path,layer,group,strength);freeze(folder/'run_config.json',plan)
        sweep.reuse_baselines(plan,folder,Path('outputs/phi4-sweeps'));task.reuse(plan,folder)
        tasks,done=task.completed(plan,folder)
        if any(k not in done for k in tasks if k[2]=='neutral'):
            raise ValueError(f'{name}: restore the existing neutral test responses; none will be regenerated.')
        result.append((plan,folder))
    return result


def worker(args):
    import torch
    from evaluate_bbh_candidates import concurrent_budget_now
    name=args.worker_task;data_path=args.data_root/name/'data_config.json';data=json.loads(data_path.read_text())
    source_plans=sources(name,data,data_path,args.output)
    concurrent_budget_now(18.)
    native=task.Native(source_plans[0][0])
    (args.output/name/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,
        transformers=task.engine.version('transformers'),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        torch_cap_gib=18.,worker_sha256=task.digest(__file__)),indent=2)+'\n')
    for plan,folder in source_plans:
        native.plan=plan
        print(f'{name}: reuse/complete collection vector L{plan["layer"]} {plan["group"]}; no standalone evaluation.',flush=True)
        task.collection_vector(plan,folder,native,args.collection_cache)
    out=args.output/name/'combined'
    with sweep.job_lock(out):
        plan,vectors=combo.prepare(out,source_plans[0][1],source_plans[1][1],dataset=name,combined_only=True)
        combo.run(plan,out,vectors,native=native.native)
    print(f'{name}: combined run complete. Use --mode summary in the CPU environment.',flush=True)


def summary(args):
    for name in DATASETS:
        out=args.output/name/'combined';path=out/'run_config.json'
        if not path.exists():print(name,'combined run not prepared yet.');continue
        print(name,flush=True);combo.report(json.loads(path.read_text()),out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('run','summary'),default='run');p.add_argument('--worker-task',choices=DATASETS)
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--data-root',type=Path,default=Path('outputs/phi4-next-tasks'))
    p.add_argument('--collection-cache',type=Path,default=Path('outputs/phi4-sweeps/collection-cache'))
    args=p.parse_args()
    if args.mode=='summary':summary(args);return
    if args.worker_task is not None:worker(args);return
    args.output.mkdir(parents=True,exist_ok=True)
    with sweep.job_lock(args.output/'launcher'):
        for name in DATASETS:
            path=args.data_root/name/'data_config.json'
            if not path.exists():task.prepare_data(name,args.data_root)
            data=json.loads(path.read_text())
            if len(data['collection'])!=125 or len(data['test'])!=125 or data.get('max_new_tokens',512)!=512:raise ValueError('Expected original125/125 BBH split and512-token budget.')
            sources(name,data,path,args.output)
        freeze(args.output/'assignment.json',dict(protocol='phi4-bbh3-bbh5-combined-v1',workers=list(DATASETS),
            interventions=[dict(layer=0,group='audio',strength=-1.),dict(layer=30,group='answer',strength=-2.)],
            evaluation='125 combined neutral-input responses per task; reuse existing neutral baseline; no standalone evaluation.'))
        with gpu_queue():
            commands=[['.venv-bbh-phi4/bin/python','-u',str(Path(__file__).resolve()),'--worker-task',name,
                '--output',str(args.output),'--data-root',str(args.data_root),'--collection-cache',str(args.collection_cache)] for name in DATASETS]
            run_workers(commands,args.output)

if __name__=='__main__':main()
