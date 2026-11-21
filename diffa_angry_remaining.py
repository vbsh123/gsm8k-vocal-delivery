"""Sequential angry-only vectors: SVAMP493, MultiArith290 and IFEval254."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import diffa_angry_agreed as audit
import diffa_bbh_angry125 as half
import diffa_happy_nightly as collection
from steer import read_rows

DATASETS=('svamp','multiarith','ifeval')
HALVES={'multiarith':290,'ifeval':254}
OUTPUT=Path('outputs/diffa-angry-only-remaining')


def prepare_svamp(data,labels,audit_root,folder):
    """Use the493 existing agreed pairs; leave the original500 test questions fixed."""
    from copy import deepcopy
    original=deepcopy(data['target']);pairs=[];bank=dict(original['bank'])
    if len(data['collection_ids'])!=500 or len(data['test_ids'])!=500:raise ValueError('Expected original500/500 SVAMP pools.')
    if set(data['collection_ids'])&set(data['test_ids']):raise ValueError('SVAMP collection/test overlap.')
    for q in data['collection_ids']:
        entries={s:data['pinned'].get(q+'|'+s) for s in ('neutral','angry')}
        if not all(entries.values()):raise ValueError(f'Missing original SVAMP collection audio: {q}.')
        neutral,angry=(entries[s]['record'] for s in ('neutral','angry'))
        label=labels.get(angry['audio_sha256'],{})
        if label.get('classification_status')!='classified':raise ValueError(f'Complete angry classification for {q}.')
        if label.get('emotion')!='angry':continue
        if (entries['neutral']['question']!=entries['angry']['question'] or
                neutral['reference_answer']!=angry['reference_answer']):raise ValueError('SVAMP paired question/reference differs.')
        pairs.append(dict(question_id=q,neutral=neutral,emotion=angry))
        bank[q]=dict(question_id=q,question=entries['neutral']['question'],answer=neutral['reference_answer'])
    if len(pairs)!=493:raise ValueError(f'Expected the agreed493 SVAMP pairs; found{len(pairs)}. Inspect audit before changing the cohort.')
    train=[p['question_id'] for p in pairs];target=dict(original,extraction_ids=train)
    plan=dict(protocol='diffa-svamp-angry-only493-v1',name='svamp',emotion='angry',original_target=original,target=target,
              source_run=original['run'],collection_ids=train,test_ids=data['test_ids'],pairs=pairs,bank=bank,
              prompt=target['prompt'],collection_prompt=target['vector_info']['collection_prompt'],seed=42,
              max_new_tokens=target['max_new_tokens'],steps=128,block_length=32,decoder='fast_dllm',temperature=0.,
              averaging='Matched forwards within each pair; token-position means; equal per-question weight.',
              classifier=audit.MODEL,classifier_receipts=[dict(question_id=p['question_id'],angry=labels[p['emotion']['audio_sha256']]) for p in pairs],
              eligible_angry_ids=train,audit_manifest_sha256=audit.winners.digest(audit_root/'cohorts.json'),
              selection='493 angry-recognized pairs inside the original500 collection pool; no neutral recognition requirement. Original500 test unchanged; seven rejected collection questions unused.',
              original_collection_ids=data['collection_ids'],original_test_ids=data['test_ids'],common_test_ids=data['test_ids'])
    folder.mkdir(parents=True,exist_ok=True);audit.winners.freeze(folder/'run_config.json',plan)
    print('svamp:493 angry-agreed collection pairs; original500 neutral test questions unchanged.',flush=True)
    return plan


def audit_and_prepare(args):
    root=args.output;aroot=root/'audit'
    if args.mode=='audit':
        # Isolate this audit from the already-running BBH experiment's audit snapshot.
        command=[sys.executable,'-u','diffa_angry_agreed.py','--mode','audit','--output',str(root),
                 '--device',args.device,'--datasets',*args.datasets]
        subprocess.run(command,check=True)
        return None
    if not (aroot/'cohorts.json').exists():aroot=Path('outputs/diffa-angry-agreed/audit')
    data=json.loads((aroot/'cohorts.json').read_text())['datasets'];labels=audit.predictions(aroot);plans={}
    for name in args.datasets:
        folder=root/name;path=folder/'run_config.json'
        if path.exists():
            plan=json.loads(path.read_text())
            if name=='svamp' and plan['protocol']!='diffa-svamp-angry-only493-v1':
                raise ValueError('This output has the older SVAMP resplit. Use a separate --output; no files overwritten.')
        elif name=='svamp':plan=prepare_svamp(data[name],labels,aroot,folder)
        else:plan=half.prepare(name,data[name],labels,aroot,folder,half=HALVES[name])
        old=plan['original_target']
        if (audit.winners.digest(old['config_file'])!=old['config_sha256'] or
                audit.winners.digest(old['vector'])!=old['vector_sha256']):raise ValueError('Original source receipts changed.')
        audit.winners.score_rows(dict(target=plan['target']),[],folder)
        plans[name]=plan
    return plans


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('audit','prepare','run','summary'),default='run')
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='summary':
        for name in args.datasets:
            folder=args.output/name;path=folder/'run_config.json'
            if not path.exists():print(name,'not prepared.');continue
            if (folder/'primary/run_config.json').exists():
                collection.report_steering(folder/'primary');half.compare_original(json.loads(path.read_text()),folder/'primary')
            else:print(name,'evaluation not started.')
        return
    with (args.output/'remaining.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plans=audit_and_prepare(args)
        if args.mode in ('audit','prepare'):return
        if shutil.disk_usage(args.output).free<600*2**20:raise RuntimeError('Need600 MiB free for aggregate checkpoints.')
        with Path('outputs/jenny-task-vectors.lock').open('a') as lock:
            print('Waiting for existing GPU jobs, then '+ ' -> '.join(args.datasets)+' sequentially.',flush=True)
            fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(30)
            from extract_vectors import load_model
            model,tokenizer,features=load_model()
            audit.winners.atomic_text(args.output/'runtime.json',json.dumps(dict(runner_sha256=audit.winners.digest(__file__),
                collector_sha256=audit.winners.digest(collection.__file__),datasets=args.datasets,model_config=model.config.to_dict()),indent=2,default=str)+'\n')
            for name,plan in plans.items():
                folder=args.output/name;collection.collect(plan,folder,model,tokenizer,features)
                source=audit.descriptor(plan,folder);job=audit.job_for(source,plan['target'],[0.,1.])
                destination=folder/'primary';destination.mkdir(exist_ok=True)
                audit.winners.freeze(destination/'run_config.json',job);audit.winners.reuse_answers(job,destination)
                half.fill_neutral(plan,job,destination,model,tokenizer,features)
                audit.winners.generate_job(job,destination,model,tokenizer,features)
                collection.report_steering(destination,tokenizer);half.compare_original(plan,destination)


if __name__=='__main__':main()
