"""Two queued workers: MultiArith/GSM and IFEval/BBH3/BBH5, L0audio-1/L11answer-1."""
import argparse
import json
from pathlib import Path

import phi4_bbh_combo_queue as sources
import phi4_gsm_combo as combo
import phi4_sweep as sweep
import phi4_task_steering as task
from phi4_queued_workers import gpu_queue,run_workers
from kimi_bbh_layer_diagnostic import freeze

OUTPUT=Path('outputs/phi4-L0audio-minus1-L11answer-minus1')
ASSIGNMENT=(('multiarith','gsm8k'),('ifeval','bbh3','bbh5'))
SETTINGS=((0,'audio',-1.),(11,'answer',-1.))


def source_plans(name,args):
    path=args.data_root/name/'data_config.json'
    if not path.exists():task.prepare_data(name,args.data_root)
    data=json.loads(path.read_text())
    expected={'multiarith':(290,290),'gsm8k':(400,869),'ifeval':(254,254),'bbh3':(125,125),'bbh5':(125,125)}[name]
    budget=1024 if name=='ifeval' else 512
    if (len(data['collection']),len(data['test']))!=expected or data.get('max_new_tokens',512)!=budget:
        raise ValueError('Expected original split and task token budget.')
    roots=(Path('outputs/phi4-ifeval-all-layers-minus1'),) if name=='ifeval' else ()
    return sources.sources(name,data,path,args.output,settings=SETTINGS,baseline_roots=roots)


def worker(args):
    from evaluate_bbh_candidates import concurrent_budget_now
    concurrent_budget_now(18.)
    native=None
    for name in ASSIGNMENT[args.worker_id]:
        plans=source_plans(name,args)
        if native is None:
            native=task.Native(plans[0][0])
            (args.output/f'runtime-worker{args.worker_id}.json').write_text(json.dumps(dict(
                worker_sha256=task.digest(__file__),combo_runner_sha256=task.digest(combo.__file__),
                hook_sha256=task.digest(task.native_hooks.__file__),datasets=ASSIGNMENT[args.worker_id],
                torch_cap_gib=18.,models_loaded=1),indent=2)+'\n')
        cache=(Path('outputs/phi4-ifeval-all-layers-minus1/collection-cache') if name=='ifeval' else args.collection_cache)
        for plan,folder in plans:
            native.plan=plan
            print(f'{name}: reuse/complete L{plan["layer"]} {plan["group"]} vector; no standalone evaluation.',flush=True)
            task.collection_vector(plan,folder,native,cache)
        out=args.output/name/'combined'
        with sweep.job_lock(out):
            plan,vectors=combo.prepare(out,plans[0][1],plans[1][1],dataset=name,combined_only=True,
                                      answer_layer=11,answer_strength=-1.,audio_strength=-1.)
            combo.run(plan,out,vectors,native=native.native)
        print(f'{name}: combined responses complete.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('run','summary'),default='run')
    p.add_argument('--worker-id',type=int,choices=(0,1))
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--data-root',type=Path,default=Path('outputs/phi4-next-tasks'))
    p.add_argument('--collection-cache',type=Path,default=Path('outputs/phi4-sweeps/collection-cache'))
    args=p.parse_args()
    if args.mode=='summary':
        for names in ASSIGNMENT:
            for name in names:
                out=args.output/name/'combined';path=out/'run_config.json'
                if path.exists():
                    print(name,flush=True);combo.report(json.loads(path.read_text()),out)
                else:print(name,'combined run not prepared yet.')
        return
    if args.worker_id is not None:worker(args);return
    args.output.mkdir(parents=True,exist_ok=True)
    with sweep.job_lock(args.output/'launcher'):
        for names in ASSIGNMENT:
            for name in names:source_plans(name,args)
        freeze(args.output/'assignment.json',dict(workers=ASSIGNMENT,settings=SETTINGS,
            note='Two parallel workers, one model per worker reused sequentially; combined-only tests, original baselines and budgets.'))
        with gpu_queue():
            commands=[['.venv-bbh-phi4/bin/python','-u',str(Path(__file__).resolve()),'--worker-id',str(i),
                       '--output',str(args.output),'--data-root',str(args.data_root),'--collection-cache',str(args.collection_cache)] for i in range(2)]
            run_workers(commands,args.output)


if __name__=='__main__':main()
