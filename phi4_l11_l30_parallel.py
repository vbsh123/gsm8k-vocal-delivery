"""Two immediate workers: L11/L30 answers, with or without L0 audio, all six tasks."""
import argparse
import json
from pathlib import Path

import phi4_bbh_combo_queue as sources
import phi4_gsm_combo as combo
import phi4_sweep as sweep
import phi4_task_steering as task
from phi4_queued_workers import run_workers
from kimi_bbh_layer_diagnostic import atomic_save,digest,freeze

OUTPUT=Path('outputs/phi4-L11-L30-double-triple')
DATASETS=('multiarith','gsm8k','svamp','bbh3','bbh5','ifeval')
COMBOS={
    'L11-L30':((11,'answer',-1.),(30,'answer',-2.)),
    'L0-L11-L30':((0,'audio',-1.),(11,'answer',-1.),(30,'answer',-2.)),
}


def source_plans(name,args):
    path=args.data_root/name/'data_config.json'
    if not path.exists():task.prepare_data(name,args.data_root)
    data=json.loads(path.read_text())
    expected={'multiarith':(290,290),'gsm8k':(400,869),'svamp':(500,500),'bbh3':(125,125),'bbh5':(125,125),'ifeval':(254,254)}[name]
    if (len(data['collection']),len(data['test']))!=expected or data.get('max_new_tokens',512)!=(1024 if name=='ifeval' else 512):
        raise ValueError('Original task split/budget differs.')
    # Both workers share these source vectors, not evaluation jobs or model instances.
    roots=(Path('outputs/phi4-ifeval-all-layers-minus1'),) if name=='ifeval' else ()
    settings=((0,'audio',-1.),(11,'answer',-1.),(30,'answer',-2.))
    return sources.sources(name,data,path,args.output,settings=settings,baseline_roots=roots)


def prepare_combo(output,plans,settings):
    import torch
    base=plans[0][0];directions={};receipts=[]
    for plan,folder in plans:
        for key in ('collection','recipe','prompt','max_new_tokens','bank','data_config_sha256'):
            if plan[key]!=base[key]:raise ValueError('Source recipes/cohorts differ: '+key)
        state=torch.load(folder/'style-vectors.pt',map_location='cpu',weights_only=True)
        if (state['run_config_sha256']!=digest(folder/'run_config.json') or
                state['collection_question_ids']!=[p['question_id'] for p in plan['collection']] or
                state['layer']!=plan['layer'] or state['group']!=plan['group'] or
                state['model']!=plan['recipe']['model'] or state['revision']!=plan['recipe']['revision']):
            raise ValueError('Source vector provenance differs.')
        direction=state['angry_minus_neutral']
        if direction.shape!=(3072,) or not torch.isfinite(direction).all():raise ValueError('Invalid raw vector.')
        directions[plan['layer'],plan['group']]=direction
        receipts.append(dict(folder=str(folder),config_sha256=digest(folder/'run_config.json'),vector_sha256=digest(folder/'style-vectors.pt')))
    rows=[r for r in base['recordings'] if r['condition']=='neutral']
    if {p['question_id'] for p in base['collection']} & {r['question_id'] for r in rows}:raise ValueError('Collection/test overlap.')
    changes=[dict(layer=layer,group=group,strength=strength) for layer,group,strength in settings]
    plan=dict(protocol='phi4-L11-L30-combination-v1',task=base['task'],recipe=base['recipe'],prompt=base['prompt'],
              seed=42,max_new_tokens=base['max_new_tokens'],collection=base['collection'],bank=base['bank'],test=rows,
              conditions={'neutral':[],'combined':changes},sources=receipts,
              note='Exploratory fixed double/triple intervention; unchanged original splits, budgets and baselines. No standalone evaluations.')
    output.mkdir(parents=True,exist_ok=True);freeze(output/'run_config.json',plan)
    path=output/'style-vectors.pt';state=dict(run_config_sha256=digest(output/'run_config.json'),directions=directions)
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=True)
        if old['run_config_sha256']!=state['run_config_sha256'] or set(old['directions'])!=set(directions) or any(
                not torch.equal(old['directions'][key],value) for key,value in directions.items()):raise ValueError('Combined vectors changed.')
    else:atomic_save(path,state)
    return plan,directions


def worker(args):
    from evaluate_bbh_candidates import concurrent_budget_now
    concurrent_budget_now(18.)
    label=list(COMBOS)[args.worker_id];settings=COMBOS[label];native=None
    for name in args.datasets:
        # Parent has frozen source plans and reused neutral baselines before starting workers.
        plans=[]
        for layer,group,_ in settings:
            folder=args.output/name/'sources'/f'L{layer}-{group}'
            plans.append((json.loads((folder/'run_config.json').read_text()),folder))
        if native is None:
            native=task.Native(plans[0][0])
            (args.output/f'runtime-worker{args.worker_id}.json').write_text(json.dumps(dict(
                combo=label,settings=settings,runner_sha256=digest(__file__),hooks_sha256=digest(task.native_hooks.__file__),
                models_loaded=1,torch_cap_gib=18.,gpu_wait=False),indent=2)+'\n')
        cache=Path('outputs/phi4-ifeval-all-layers-minus1/collection-cache') if name=='ifeval' else args.collection_cache
        for plan,folder in plans:
            native.plan=plan
            print(f'{label}/{name}: reuse/complete L{plan["layer"]} {plan["group"]} vector.',flush=True)
            task.collection_vector(plan,folder,native,cache)
        out=args.output/name/label
        with sweep.job_lock(out):
            plan,vectors=prepare_combo(out,plans,settings)
            combo.run(plan,out,vectors,native=native.native)
        print(f'{label}/{name}: complete.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('run','summary'),default='run')
    p.add_argument('--worker-id',type=int,choices=(0,1))
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--data-root',type=Path,default=Path('outputs/phi4-next-tasks'))
    p.add_argument('--collection-cache',type=Path,default=Path('outputs/phi4-sweeps/collection-cache'))
    args=p.parse_args()
    if args.mode=='summary':
        for name in args.datasets:
            for label in COMBOS:
                out=args.output/name/label;path=out/'run_config.json'
                print(name,label,flush=True)
                if path.exists():combo.report(json.loads(path.read_text()),out)
                else:print('Not prepared yet.')
        return
    if args.worker_id is not None:worker(args);return
    args.output.mkdir(parents=True,exist_ok=True)
    with sweep.job_lock(args.output/'launcher'):
        for name in args.datasets:source_plans(name,args)
        freeze(args.output/'assignment.json',dict(workers=COMBOS,datasets=args.datasets,
               policy='Start two workers immediately; one persistent model each; no shared GPU queue or idle-GPU wait.'))
        commands=[['.venv-bbh-phi4/bin/python','-u',str(Path(__file__).resolve()),'--worker-id',str(i),
                   '--output',str(args.output),'--data-root',str(args.data_root),'--collection-cache',str(args.collection_cache),
                   '--datasets',*args.datasets] for i in range(2)]
        run_workers(commands,args.output)


if __name__=='__main__':main()
