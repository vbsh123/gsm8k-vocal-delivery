"""Queued SVAMP500 combined-only L0 audio -1 + L11 answer -1."""
import argparse
import json
from pathlib import Path

import phi4_bbh_combo_queue as collection
import phi4_gsm_combo as combo
import phi4_sweep as sweep
import phi4_task_steering as task
from phi4_queued_workers import gpu_queue
from phi4_svamp_layers_parallel import collection_config
from kimi_bbh_layer_diagnostic import atomic_save,digest,freeze

OUTPUT=Path('outputs/phi4-svamp-L0audio-minus1-L11answer-minus1')
SETTINGS=((0,'audio',-1.),(11,'answer',-1.))


def reuse_subset_pairs(plan,cache_root,source_cache):
    """Reuse individual compatible300-cohort captures, not its300-question mean vector."""
    import fcntl
    import torch
    source_config=source_cache/'run_config.json'
    if not source_config.exists():return 0
    original=json.loads(source_config.read_text());common=collection_config(plan)
    if any(original.get(k)!=common[k] for k in common if k!='collection'):return 0
    cache=cache_root/plan['task']/f"L{plan['layer']}-{plan['group']}";cache.mkdir(parents=True,exist_ok=True)
    with (cache/'collection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        freeze(cache/'run_config.json',common);(cache/'pairs').mkdir(exist_ok=True)
        previous={p['question_id']:(i,p) for i,p in enumerate(original['collection'])};count=0
        for i,pair in enumerate(plan['collection']):
            destination=cache/'pairs'/f'{i:04d}.pt';item=previous.get(pair['question_id'])
            if destination.exists() or item is None:continue
            j,old_pair=item;path=source_cache/'pairs'/f'{j:04d}.pt'
            if not path.exists():continue
            saved=torch.load(path,map_location='cpu',weights_only=True)
            task.engine.validate_pair(saved,old_pair,digest(source_config),original['protocol'])
            if saved['voice']!=pair['voice'] or saved['audio_sha256']!={s:pair[s]['audio_sha256'] for s in ('neutral','angry')}:continue
            copied=dict(saved,run_config_sha256=digest(cache/'run_config.json'),
                reused_pair=dict(path=str(path),sha256=digest(path)))
            task.engine.validate_pair(copied,pair,digest(cache/'run_config.json'),common['protocol'])
            atomic_save(destination,copied);count+=1
        print(f'Reused{count} compatible L{plan["layer"]} {plan["group"]} collection pairs; complete remaining pairs to500.',flush=True)
        return count


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('run','summary'),default='run')
    p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--data-root',type=Path,default=Path('outputs/phi4-next-tasks'))
    p.add_argument('--collection-cache',type=Path,default=Path('outputs/phi4-sweeps/collection-cache'))
    args=p.parse_args();out=args.output/'svamp/combined'
    if args.mode=='summary':
        path=out/'run_config.json'
        if not path.exists():print('SVAMP L0−1/L11−1 combined run not prepared yet.');return
        combo.report(json.loads(path.read_text()),out);return
    args.output.mkdir(parents=True,exist_ok=True)
    with sweep.job_lock(args.output/'launcher'):
        path=args.data_root/'svamp/data_config.json'
        if not path.exists():task.prepare_data('svamp',args.data_root)
        data=json.loads(path.read_text())
        if len(data['collection'])!=500 or len(data['test'])!=500:raise ValueError('Expected existing disjoint500/500 SVAMP split.')
        plans=collection.sources('svamp',data,path,args.output,settings=SETTINGS)
        with gpu_queue():
            from evaluate_bbh_candidates import concurrent_budget_now
            concurrent_budget_now(18.)
            native=task.Native(plans[0][0])
            for plan,folder in plans:
                source=Path('outputs/phi4-sweeps/svamp300-300')
                if plan['group']=='audio':source=Path('outputs/phi4-sweeps/svamp300-300-audio')
                reuse_subset_pairs(plan,args.collection_cache,source/'collection-cache/svamp'/f"L{plan['layer']}-{plan['group']}")
                native.plan=plan;task.collection_vector(plan,folder,native,args.collection_cache)
            with sweep.job_lock(out):
                plan,vectors=combo.prepare(out,plans[0][1],plans[1][1],dataset='svamp',combined_only=True,
                    answer_layer=11,answer_strength=-1.,audio_strength=-1.)
                combo.run(plan,out,vectors,native=native.native)

if __name__=='__main__':main()
