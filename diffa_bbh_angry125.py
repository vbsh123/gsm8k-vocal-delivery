"""BBH3/5: 125 angry-recognized collection pairs, evaluate the remaining125."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

import diffa_angry_agreed as agreed
import diffa_happy_nightly as collection
from diffa_collection_report import collection_bank
from steer import read_rows

TASKS=('bbh3','bbh5')


def select(data,labels,half=125):
    pool=data['collection_ids']+data['test_ids']
    if len(pool)!=2*half or len(set(pool))!=2*half:raise ValueError(f'Expected original{2*half} unique questions.')
    pairs={};eligible=set()
    for q in pool:
        entries={s:data['pinned'].get(q+'|'+s) for s in ('neutral','angry')}
        if not all(entries.values()):raise ValueError(f'Restore original paired audio for {q}.')
        neutral,angry=(entries[s]['record'] for s in ('neutral','angry'))
        if (entries['neutral']['question']!=entries['angry']['question'] or
                neutral['reference_answer']!=angry['reference_answer']):raise ValueError('Paired question/reference differs.')
        label=labels.get(angry['audio_sha256'],{})
        if label.get('classification_status')!='classified':raise ValueError(f'Complete angry classification for {q}.')
        pairs[q]=dict(question_id=q,neutral=neutral,emotion=angry)
        if label.get('emotion')=='angry':eligible.add(q)
    if len(eligible)<half:raise ValueError(f'Only{len(eligible)} angry-recognized pairs; need{half}.')
    # Retain eligible original training IDs first to maximize the common held-out comparison.
    rng=random.Random(42)
    first=sorted(eligible & set(data['collection_ids']));extra=sorted(eligible & set(data['test_ids']))
    rng.shuffle(first);rng.shuffle(extra)
    train=(first+extra)[:half];test=[q for q in pool if q not in set(train)]
    return pairs,train,test


def prepare(name,data,labels,audit_root,folder,half=125):
    pairs,train,test=select(data,labels,half)
    original=deepcopy(data['target']);bank=dict(original['bank'])
    for q,pair in pairs.items():
        bank[q]=dict(bank.get(q,{}),question_id=q,question=data['pinned'][q+'|neutral']['question'],answer=pair['neutral']['reference_answer'])
    metadata=dict(name=name,collection_ids=list(pairs),pairs=list(pairs.values()),bank=bank,target=original)
    bank=collection_bank(metadata)
    target=dict(original,records=[pairs[q]['neutral'] for q in test],bank={q:bank[q] for q in test},
                extraction_ids=train)
    protocol='diffa-bbh-angry-only125-v1' if name in TASKS else f'diffa-{name}-angry-only-half-v1'
    plan=dict(protocol=protocol,name=name,emotion='angry',original_target=original,target=target,
              source_run=original['run'],collection_ids=train,test_ids=test,pairs=[pairs[q] for q in train],bank=bank,
              prompt=target['prompt'],collection_prompt=target['vector_info']['collection_prompt'],seed=42,
              max_new_tokens=target.get('max_new_tokens',512),steps=128,block_length=32,decoder='fast_dllm',temperature=0.,
              averaging='Matched forwards within each pair; token-position means; equal per-question weight.',
              classifier=agreed.MODEL,classifier_receipts=[dict(question_id=q,angry=labels[pairs[q]['emotion']['audio_sha256']]) for q in train],
              eligible_angry_ids=[q for q,pair in pairs.items() if labels[pair['emotion']['audio_sha256']]['emotion']=='angry'],
              audit_manifest_sha256=agreed.winners.digest(audit_root/'cohorts.json'),
              selection=f'Angry label only; neutral recognition ignored. Seed42; prefer eligible original collection IDs, then fill from eligible original test IDs. Remainder{half} is new test.',
              original_collection_ids=data['collection_ids'],original_test_ids=data['test_ids'],
              common_test_ids=[q for q in test if q in set(data['test_ids'])],
              note='Exploratory new split after prior results. Raw old test cases used for collection are excluded from the new test. Compare old/new vectors only on common held-out IDs.')
    folder.mkdir(parents=True,exist_ok=True);agreed.winners.freeze(folder/'run_config.json',plan)
    print(f'{name}:{half} angry-agreed pairs,{half} new test; common original held-out={len(plan["common_test_ids"])}.',flush=True)
    return plan


def fill_neutral(plan,job,folder,model,tokenizer,features):
    done={r['question_id'] for r in read_rows(folder/'results.jsonl') if r['condition']=='neutral'}
    pending=[r for r in job['target']['records'] if r['question_id'] not in done]
    print(f'{plan["name"]}: {len(done)} neutral answers reused; {len(pending)} missing new-test baselines.',flush=True)
    with (folder/'results.jsonl').open('a') as stream:
        for record in pending:
            response,encoding=collection.generate(record,plan,plan['prompt'],model,tokenizer,features)
            row=dict(record,response=response,condition='neutral',strength=0.,prompt=plan['prompt'],seed=42,
                     max_new_tokens=plan.get('max_new_tokens',512),steps=128,block_length=32,temperature=0.,decoder='fast_dllm',interventions=[],
                     audio_encoding=encoding,response_tokens=len(tokenizer.encode(response,add_special_tokens=False)))
            stream.write(json.dumps(row)+'\n');stream.flush()


def compare_original(plan,folder):
    """Direct paired comparison on questions held out from BOTH collections."""
    from rescore_audio_steering import paired
    old=plan['original_target'];original=Path(old['run']);config=json.loads(agreed.winners.config_file(original).read_text())
    wanted=set(plan['common_test_ids']);records={r['question_id']:r for r in plan['target']['records']}
    rows=[]
    for row in read_rows(original/'results.jsonl'):
        q=row['question_id']
        if (q in wanted and agreed.winners.saved_strength(row,config,old)==1.
                and agreed.winners.reusable(row,config,records[q],plan['target'])):
            rows.append(dict(row,condition='original-vector'))
    for row in read_rows(folder/'results.jsonl'):
        if row['question_id'] in wanted and row['strength'] in (0.,1.):
            rows.append(dict(row,condition='neutral' if row['strength']==0 else 'new-vector'))
    destination=folder/'common-heldout';destination.mkdir(exist_ok=True)
    scored,provenance=agreed.winners.score_rows(dict(target=plan['target']),rows,destination)
    groups={c:{r['question_id']:r for r in scored if r['condition']==c} for c in ('neutral','original-vector','new-vector')}
    ids=sorted(set.intersection(*(set(g) for g in groups.values())));results=[]
    print(f'{plan["name"]}: {len(ids)}/{len(wanted)} questions held out from both vectors.',flush=True)
    if ids:
        for metric in groups['neutral'][ids[0]]['metrics']:
            counts={c:sum(g[q]['metrics'][metric] for q in ids) for c,g in groups.items()}
            result=paired([(groups['original-vector'][q]['metrics'][metric],groups['new-vector'][q]['metrics'][metric]) for q in ids])
            results.append(dict(metric=metric,counts=counts,**result))
            print(f'{metric}: neutral={counts["neutral"]}/{len(ids)}, original={counts["original-vector"]}/{len(ids)}, '
                  f'new={counts["new-vector"]}/{len(ids)}; new vs original gain={result["gain_pp"]:+.2f}; p={result["mcnemar_exact_p"]:.4g}',flush=True)
    agreed.winners.atomic_text(destination/'summary.json',json.dumps(dict(question_ids=ids,paired=results,scorer=provenance),indent=2)+'\n')
    agreed.winners.atomic_text(destination/'rescored.jsonl',''.join(json.dumps(r)+'\n' for r in scored))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','run','summary'),default='run')
    p.add_argument('--datasets',nargs='+',choices=TASKS,default=list(TASKS))
    p.add_argument('--audit-root',type=Path,default=Path('outputs/diffa-angry-agreed/audit'))
    p.add_argument('--output',type=Path,default=Path('outputs/diffa-bbh-angry-only125'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='summary':
        for name in args.datasets:
            folder=args.output/name;path=folder/'run_config.json'
            if not path.exists():print(name,'not prepared.');continue
            if (folder/'primary/run_config.json').exists():
                collection.report_steering(folder/'primary')
                compare_original(json.loads(path.read_text()),folder/'primary')
            else:print(name,'evaluation not started.')
        return
    with (args.output/'queue.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        data=json.loads((args.audit_root/'cohorts.json').read_text())['datasets'];labels=agreed.predictions(args.audit_root)
        plans={}
        for name in args.datasets:
            folder=args.output/name;path=folder/'run_config.json'
            if path.exists():plan=json.loads(path.read_text())
            else:plan=prepare(name,data[name],labels,args.audit_root,folder)
            old=plan['original_target']
            if (agreed.winners.digest(old['config_file'])!=old['config_sha256'] or
                    agreed.winners.digest(old['vector'])!=old['vector_sha256']):raise ValueError('Original source changed.')
            plans[name]=plan
            agreed.winners.score_rows(dict(target=plan['target']),[],folder)  # Official evaluator preflight.
        if args.mode=='prepare':return
        if shutil.disk_usage(args.output).free<600*2**20:raise RuntimeError('Need600 MiB free for aggregate checkpoints.')
        with Path('outputs/jenny-task-vectors.lock').open('a') as lock:
            print('Waiting for shared DIFFA GPU queue and idle GPU.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(30)
            from extract_vectors import load_model
            model,tokenizer,features=load_model()
            agreed.winners.atomic_text(args.output/'runtime.json',json.dumps(dict(
                runner_sha256=agreed.winners.digest(__file__),collector_sha256=agreed.winners.digest(collection.__file__),
                datasets=args.datasets,model_config=model.config.to_dict()),indent=2,default=str)+'\n')
            for name,plan in plans.items():
                folder=args.output/name;collection.collect(plan,folder,model,tokenizer,features)
                source=agreed.descriptor(plan,folder);job=agreed.job_for(source,plan['target'],[0.,1.])
                destination=folder/'primary';destination.mkdir(exist_ok=True)
                agreed.winners.freeze(destination/'run_config.json',job);agreed.winners.reuse_answers(job,destination)
                fill_neutral(plan,job,destination,model,tokenizer,features)
                agreed.winners.generate_job(job,destination,model,tokenizer,features)
                collection.report_steering(destination,tokenizer);compare_original(plan,destination)


if __name__=='__main__':main()
