"""Agreed-only overnight emotions: shared bare prompts/cohorts, DIFFA block collection, Phi three-layer collection."""
import argparse
from copy import deepcopy
import fcntl
import itertools
import json
import os
from pathlib import Path
import random
from statistics import mean,median
import subprocess
import time

import torch
import diffa_vanilla_all as catalog
import diffa_gsm_vanilla as vanilla
import diffa_active_pooling as blocks
import diffa_stream_collection as streaming
from jenny_bundle_registry import Registry,STYLES
from steer import read_rows

winners=vanilla.task.winners
EMOTIONS=('angry','happy','sad')
SPECS=dict(gsm8k=(1319,425),**catalog.SPECS)
PROTOCOL='emotion-night-bare-agreed-block-v1'
PHI_TARGETS={'L0-audio':(0,'audio'),'L11-answer':(11,'answer'),'L30-answer':(30,'answer')}


def audio_root(explicit=None):
    if explicit is not None:
        return explicit
    repo=Path(__file__).resolve().parent
    for path in (repo/'outputs/jenny-zip-audio',repo):
        if (path/'metadata/audio_inventory.jsonl').is_file() or (path/'outputs/shared-three-emotions/audio_inventory.jsonl').is_file():
            return path
    raise FileNotFoundError('Extract the existing Jenny audio ZIP, or supply --audio-root with its directory.')


def cohort(name,registry,spec=None):
    total,cap=spec or SPECS[name]
    bank,records,labels={},{},{}
    pending=[]
    for q in registry.neutral_ids(name):
        neutral=registry.choose(name,q,'neutral')
        for style in STYLES:
            item=registry.choose(name,q,style,question=neutral['question'],answer=neutral['answer'])
            if item is None:
                raise ValueError(f'Missing {name}/{q}/{style}; restore complete Jenny audio.')
            variants={r['record']['audio_sha256'] for r in registry.options[name,q,style]
                      if r['question']==neutral['question'] and r['answer']==neutral['answer']}
            if len(variants)!=1:
                raise ValueError(f'Ambiguous recordings: {name}/{q}/{style}.')
            records[q,style]=item['record']
            if style!='neutral':
                label=registry.labels.get(item['record']['audio_sha256'],{})
                if label.get('classification_status')!='classified':
                    pending.append((q,style))
                    labels[q,style]=None
                else:
                    labels[q,style]=label['emotion']
        bank[q]=dict(question_id=q,question=neutral['question'],answer=neutral['answer'])
    if len(bank)!=total:
        raise ValueError(f'{name}: expected {total} questions; found {len(bank)}.')
    if pending:
        print(f'{name}: {len(pending)} unclassified emotion recordings remain in natural evaluation and are EXCLUDED from fitting; examples={pending[:3]}.',flush=True)
    bank=catalog.scoring_bank(name,bank,records)
    order=sorted(bank)
    random.Random(42).shuffle(order)
    fit={}
    for emotion in EMOTIONS:
        eligible=sorted(q for q in order if labels[q,emotion]==emotion)
        if not eligible:
            raise ValueError(f'{name}/{emotion}: no e2v-agreed fitting clips.')
        selected=set(random.Random(44).sample(eligible,min(cap,len(eligible))))
        fit[emotion]=[q for q in order if q in selected]
    manifest=dict(protocol=PROTOCOL,name=name,prompt=catalog.prompt_for(name),
        max_new_tokens=1024 if name=='ifeval' else 512,seed=42,question_order=order,bank=bank,
        fitting_ids=fit,fitting_cap=cap,
        audio_sha256={q:{s:records[q,s]['audio_sha256'] for s in STYLES} for q in sorted(bank)},
        predictions={q:{e:labels[q,e] for e in EMOTIONS} for q in sorted(bank)},
        selection='Each emotion independently: seed44 sample min(cap, all e2v-agreed clips). Neutral prediction ignored. No response filtering. All questions receive all four natural deliveries.')
    return manifest,records


def prepare(name,registry,root,model,spec=None):
    manifest,records=cohort(name,registry,spec)
    folder=root/model/name
    folder.mkdir(parents=True,exist_ok=True)
    winners.freeze(folder/'cohort.json',manifest)
    checksum=vanilla.fingerprint(manifest)
    if model=='phi':
        import evaluate_gsm_candidates as gsm
        model_recipe=dict(gsm.MODELS['phi4'],dtype='bfloat16',do_sample=False,num_beams=1,use_cache=True,repetition_penalty=1.)
    else:
        model_recipe=dict(vanilla.RECIPE,max_new_tokens=manifest['max_new_tokens'])
    plan=dict(protocol=PROTOCOL,model=model,name=name,task=name,cohort_sha256=checksum,
        prompt=manifest['prompt'],collection_prompt=manifest['prompt'],bank=manifest['bank'],
        max_new_tokens=manifest['max_new_tokens'],seed=42,recipe=model_recipe,
        question_order=manifest['question_order'],fitting_ids=manifest['fitting_ids'],
        records=[dict(records[q,s],condition=s) for q in manifest['question_order'] for s in STYLES],
        collection=('Current32-token answer-block means, matched chronological forward prefixes, equal question weight.' if model=='diffa' else
                    'One native generation captures L0 audio first prefill and L11/L30 last-answer states averaged over all native forwards, including prefill.'),
        steering=('L31 answer +1 on every neutral outside each emotion fitting set.' if model=='diffa' else 'None; evaluation and three-layer collection only.'))
    winners.freeze(folder/'run_config.json',plan)
    emotion_plans={}
    for e in EMOTIONS:
        ids=plan['fitting_ids'][e]
        p=dict(protocol=PROTOCOL,name=name,emotion=e,collection_ids=ids,prompt=plan['prompt'],
            collection_prompt=plan['prompt'],max_new_tokens=plan['max_new_tokens'],seed=42,
            bank=plan['bank'],pairs=[dict(question_id=q,neutral=records[q,'neutral'],emotion=records[q,e]) for q in ids],
            averaging=plan['collection'],model=model,cohort_sha256=checksum)
        winners.freeze(folder/e/'run_config.json',p)
        emotion_plans[e]=p
    print(f'{model}/{name}: COHORT SHA256 {checksum}; {len(manifest["question_order"])} questions × four natural deliveries.',flush=True)
    print('Prompt:',repr(plan['prompt']),'Budget:',plan['max_new_tokens'],flush=True)
    print('Collection:',plan['collection'],flush=True)
    print('Steering:',plan['steering'],flush=True)
    for e in EMOTIONS:
        agreed=sum(manifest['predictions'][q][e]==e for q in manifest['question_order'])
        unknown=sum(manifest['predictions'][q][e] is None for q in manifest['question_order'])
        print(f'  {e}: e2v-agreed={agreed}; unclassified={unknown}; collect={len(plan["fitting_ids"][e])}; steering holdout={len(manifest["question_order"])-len(plan["fitting_ids"][e])}.',flush=True)
    return folder,plan,emotion_plans


def raw_rows(folder,plan,repair=False):
    expected={(r['question_id'],r['condition']):r for r in plan['records']}
    checksum=winners.digest(folder/'run_config.json')
    done={}
    for row in read_rows(folder/'deliveries.jsonl',repair=repair):
        key=row['question_id'],row['condition']
        record=expected.get(key)
        if (key in done or record is None or row.get('run_config_sha256')!=checksum
                or row.get('prompt')!=plan['prompt'] or row.get('max_new_tokens')!=plan['max_new_tokens']
                or row.get('model')!=plan['model'] or row.get('audio_sha256')!=record['audio_sha256']
                or row.get('seed')!=42 or row.get('voice')!=record['voice']
                or row.get('reference_answer')!=record['reference_answer']
                or row.get('strength')!=0. or row.get('interventions')!=[]):
            raise ValueError('Natural response cache differs from frozen model/prompt/cohort/audio.')
        done[key]=row
    return done


class DiffaBackend:
    def __init__(self):
        from extract_vectors import load_model
        self.model,self.tokenizer,self.features=load_model()
    def generate(self,record,plan,capture=False):
        if not capture:
            response,encoding=vanilla.task.generate(record,plan,plan['prompt'],self.model,self.tokenizer,self.features)
            return dict(response=response,response_tokens=len(self.tokenizer.encode(response,add_special_tokens=False)),audio_encoding=encoding)
        import src.fast_dllm.diffa_generate as decoder
        response,encoding,full,active=blocks.capture_poolings(self.model,
            lambda:vanilla.task.generate(record,plan,plan['prompt'],self.model,self.tokenizer,self.features),decoder,plan['max_new_tokens'])
        return dict(response=response,response_tokens=len(self.tokenizer.encode(response,add_special_tokens=False)),
                    audio_encoding=encoding,full=full,active=active,forwards=full.shape[1])


class PhiBackend:
    def __init__(self):
        from steer_phi4_gsm_reverse import NativePhi
        self.native=NativePhi(11)
    def generate(self,record,plan,capture=False):
        import phi4_all_layer_diagnostic as hooks
        trace=hooks.generate(self.native,record,plan['prompt'],collect=capture,
                             capture_layers=[0,11,30] if capture else None,max_new_tokens=plan['max_new_tokens'])
        if capture:
            values=trace.pop('means')
            if values.shape!=(3,2,3072) or not torch.isfinite(values).all():
                raise ValueError('Missing Phi L0/L11/L30 capture from the same generation.')
            trace['selected_means']=torch.stack([values[0,1],values[1,0],values[2,0]])
            trace['forwards']=trace['hook_forwards']
        return trace


def load_state(p,folder):
    if p['model']=='diffa':
        return streaming.load(p,folder,winners.digest(folder/'run_config.json'))
    path=folder/'collection-stream.pt'
    checksum=winners.digest(folder/'run_config.json')
    state=torch.load(path,map_location='cpu',weights_only=True) if path.exists() else dict(
        run_config_sha256=checksum,completed_ids=[],rows=[],difference_sum=None)
    prefix=p['collection_ids'][:len(state['completed_ids'])]
    if state['run_config_sha256']!=checksum or state['completed_ids']!=prefix or [r['question_id'] for r in state['rows']]!=prefix:
        raise ValueError('Phi checkpoint differs from the exact fitting prefix.')
    return state


def append_pair(p,state,q,n,e):
    if p['model']=='diffa':
        pooled,matched=blocks.pool_pair(n['full'],e['full'],n['active'],e['active'])
        nm,em,delta=pooled['active-block']
        streaming.append(state,dict(question_id=q,neutral=nm,emotion=em,difference=delta,
            neutral_response=n['response'],emotion_response=e['response'],neutral_forwards=n['forwards'],
            emotion_forwards=e['forwards'],matched_forwards=matched,
            neutral_encoding=n['audio_encoding'],emotion_encoding=e['audio_encoding']))
    else:
        delta=e['selected_means']-n['selected_means']
        if delta.shape!=(3,3072) or not torch.isfinite(delta).all():
            raise ValueError('Invalid three-direction Phi difference.')
        value=delta.double()
        state['difference_sum']=value.clone() if state['difference_sum'] is None else state['difference_sum']+value
        state['completed_ids'].append(q)
        state['rows'].append(dict(question_id=q,neutral_response=n['response'],emotion_response=e['response'],
            neutral_forwards=n['forwards'],emotion_forwards=e['forwards']))


def finalize(p,folder,state):
    if state['completed_ids']!=p['collection_ids']:
        raise ValueError('Collection incomplete.')
    if p['model']=='diffa':
        streaming.finalize(p,folder,state)
        return
    direction=state['difference_sum'].float()/len(p['collection_ids'])
    vector=dict(protocol=PROTOCOL,emotion=p['emotion'],sign=f'{p["emotion"]} minus neutral',
        targets=PHI_TARGETS,collection_question_ids=p['collection_ids'],prompt=p['prompt'],
        max_new_tokens=p['max_new_tokens'],directions={k:direction[i] for i,k in enumerate(PHI_TARGETS)},
        run_config_sha256=winners.digest(folder/'run_config.json'),pairs=len(p['collection_ids']))
    path=folder/'style-vectors.pt'
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=True)
        if old['run_config_sha256']!=vector['run_config_sha256'] or any(not torch.equal(old['directions'][k],v) for k,v in vector['directions'].items()):
            raise ValueError('Phi recomputed vector differs.')
    else:
        vanilla.task.save_tensor(torch,vector,path)


def generate_dataset(folder,plan,plans,backend):
    done=raw_rows(folder,plan,repair=True)
    states={e:load_state(p,folder/e) for e,p in plans.items()}
    members={e:set(p['collection_ids']) for e,p in plans.items()}
    records={(r['question_id'],r['condition']):r for r in plan['records']}
    checksum=winners.digest(folder/'run_config.json')
    with (folder/'deliveries.jsonl').open('a') as stream:
        def save(q,style,trace):
            key=q,style
            if key in done:
                if done[key]['response']!=trace['response']:
                    raise ValueError(f'Replayed collection response changed: {key}')
                return
            record=records[key]
            row=dict(record,response=trace['response'],response_tokens=trace['response_tokens'],
                model=plan['model'],run_config_sha256=checksum,prompt=plan['prompt'],seed=42,
                max_new_tokens=plan['max_new_tokens'],strength=0.,interventions=[],
                **({k:plan['recipe'][k] for k in ('decoder','steps','block_length','temperature')} if plan['model']=='diffa' else {}))
            stream.write(json.dumps(row)+'\n');stream.flush();done[key]=row
        for index,q in enumerate(plan['question_order']):
            pending=[e for e in EMOTIONS if q in members[e] and q not in states[e]['completed_ids']]
            missing=[s for s in STYLES if (q,s) not in done]
            if not pending and not missing:
                continue
            print(f'{plan["model"]}/{plan["name"]} [{index+1}/{len(plan["question_order"])}] {q}; capture={pending}; missing={missing}',flush=True)
            for e in pending:
                if plans[e]['collection_ids'][len(states[e]['completed_ids'])]!=q:
                    raise ValueError('Traversal differs from frozen fitting prefix.')
            n=None
            if pending or 'neutral' in missing:
                n=backend.generate(records[q,'neutral'],plan,capture=bool(pending))
                save(q,'neutral',n)
            for e in EMOTIONS:
                if e in pending:
                    trace=backend.generate(records[q,e],plan,capture=True)
                    save(q,e,trace)
                    append_pair(plans[e],states[e],q,n,trace)
                    vanilla.task.save_tensor(torch,states[e],folder/e/'collection-stream.pt')
                    del trace
                elif e in missing:
                    trace=backend.generate(records[q,e],plan,capture=False)
                    save(q,e,trace)
                    del trace
            del n
    for e in EMOTIONS:
        finalize(plans[e],folder/e,states[e])


def natural_report(folder,plan):
    from rescore_audio_steering import paired
    rows=list(raw_rows(folder,plan).values())
    scored,provenance=winners.score_rows(dict(target=plan),rows,folder/'scoring')
    groups={s:{r['question_id']:r for r in scored if r['condition']==s} for s in STYLES}
    print(f'\n{plan["model"]}/{plan["name"]}: natural deliveries {len(rows)}/{4*len(plan["question_order"])}; cohort={plan["cohort_sha256"]}',flush=True)
    lengths={}
    instruction_totals={}
    for s,values in groups.items():
        if not values:
            continue
        tokens=[r['response_tokens'] for r in values.values()]
        lengths[s]=dict(n=len(tokens),mean_tokens=mean(tokens),median_tokens=median(tokens))
        print(f'{s}: N={len(values)}; mean/median tokens={mean(tokens):.1f}/{median(tokens):g}',flush=True)
        for metric in next(iter(values.values()))['metrics']:
            print(f'  {metric}: {sum(r["metrics"][metric] for r in values.values())}/{len(values)}',flush=True)
        if plan['name']=='ifeval':
            instruction_totals[s]={}
            for mode in ('strict','loose'):
                bits=[bit for row in values.values() for bit in row['official_scores'][mode]['instructions']]
                instruction_totals[s][mode]=dict(correct=sum(bits),total=len(bits),accuracy=sum(bits)/len(bits) if bits else None)
                print(f'  {mode} instruction: {sum(bits)}/{len(bits)}',flush=True)
    comparisons=[]
    for a,b in itertools.combinations(STYLES,2):
        ids=sorted(groups[a].keys()&groups[b].keys())
        if not ids:
            continue
        for metric in groups[a][ids[0]]['metrics']:
            value=paired([(groups[a][q]['metrics'][metric],groups[b][q]['metrics'][metric]) for q in ids])
            comparisons.append(dict(reference=a,test=b,metric=metric,**value))
            print(f'{b} vs {a} / {metric}: N={len(ids)}; gain={value["gain_pp"]:+.2f}; p={value["mcnemar_exact_p"]:.4g}; CI={value["ci95_gain_pp"]}',flush=True)
    winners.atomic_text(folder/'delivery-summary.json',json.dumps(dict(lengths=lengths,instruction_totals=instruction_totals,comparisons=comparisons,scorer=provenance,
        note='All natural questions include fitting examples; steering reports exclude fitting questions.'),indent=2)+'\n')
    winners.atomic_text(folder/'delivery-rescored.jsonl',''.join(json.dumps(r)+'\n' for r in scored))


def steering_job(folder,plan,p):
    excluded=set(p['collection_ids'])
    records=[r for r in plan['records'] if r['condition']=='neutral' and r['question_id'] not in excluded]
    source=dict(name=plan['name'],task=plan['name'],emotion=p['emotion'],
        vector=str((folder/p['emotion']/'style-vectors.pt').resolve()),prompt=plan['prompt'],
        bank=plan['bank'],records=records,extraction_ids=p['collection_ids'],**plan['recipe'])
    source['vector_sha256']=winners.digest(source['vector'])
    _,source['vector_info']=winners.read_vector(source['vector'],p['emotion'])
    return dict(protocol=PROTOCOL,emotion=p['emotion'],source=source,target=source,
        strengths=[0.,1.],layer=31,group='answer',mode='fixed',schedule='always')


def steer_dataset(folder,plan,plans,backend):
    if plan['model']!='diffa':
        raise ValueError('Phi has no steering stage in this queue.')
    from diffa_layer_probe import validate
    natural=raw_rows(folder,plan)
    for e,p in plans.items():
        job=steering_job(folder,plan,p)
        out=folder/e/'L31-answer-a1'
        winners.freeze(out/'run_config.json',job)
        path=out/'results.jsonl'
        if not path.exists():
            with path.open('x') as stream:
                for record in job['target']['records']:
                    stream.write(json.dumps(dict(natural[record['question_id'],'neutral'],vector_sha256=None))+'\n')
        validate(job,out)
        winners.generate_job(job,out,backend.model,backend.tokenizer,backend.features)
        validate(job,out)
        vanilla.task.report_steering(out,backend.tokenizer)
    steering_comparisons(folder,plan)


def steering_comparisons(folder,plan):
    """Compare emotion vectors only on questions held out from BOTH fitting sets."""
    from rescore_audio_steering import paired
    rows=[]
    for e in EMOTIONS:
        for row in read_rows(folder/e/'L31-answer-a1/results.jsonl'):
            if row['strength']==1.:
                if row['question_id'] in plan['fitting_ids'][e]:
                    raise ValueError('Fitting example entered steering comparison.')
                rows.append(dict(row,vector_emotion=e))
    if not rows:
        return
    scored,provenance=winners.score_rows(dict(target=plan),rows,folder/'steering-comparison-scoring')
    groups={e:{r['question_id']:r for r in scored if r['vector_emotion']==e} for e in EMOTIONS}
    comparisons=[]
    for a,b in itertools.combinations(EMOTIONS,2):
        ids=sorted(groups[a].keys()&groups[b].keys())
        if not ids:
            continue
        for metric in groups[a][ids[0]]['metrics']:
            value=paired([(groups[a][q]['metrics'][metric],groups[b][q]['metrics'][metric]) for q in ids])
            comparisons.append(dict(reference=a,test=b,metric=metric,**value))
            print(f'Steering {b} vs {a} / {metric}: SAME held-out N={len(ids)}; gain={value["gain_pp"]:+.2f}; p={value["mcnemar_exact_p"]:.4g}; CI={value["ci95_gain_pp"]}',flush=True)
    winners.atomic_text(folder/'steering-pairwise-summary.json',json.dumps(dict(comparisons=comparisons,scorer=provenance),indent=2)+'\n')


def report(folder,plan,plans):
    natural_report(folder,plan)
    for e,p in plans.items():
        state=load_state(p,folder/e)
        print(f'{plan["model"]}/{plan["name"]}/{e}: collected {len(state["completed_ids"])}/{len(p["collection_ids"])}.',flush=True)
        path=folder/e/'L31-answer-a1'
        if plan['model']=='diffa' and (path/'results.jsonl').exists():
            vanilla.task.report_steering(path)
    if plan['model']=='diffa':
        steering_comparisons(folder,plan)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=('diffa','phi'),default='diffa')
    parser.add_argument('--datasets',choices=tuple(SPECS),nargs='+',required=True)
    parser.add_argument('--mode',choices=('prepare','run','summary'),default='run')
    parser.add_argument('--audio-root',type=Path)
    parser.add_argument('--output',type=Path,default=Path('outputs/emotion-block-night'))
    args=parser.parse_args()
    if args.model=='phi' and args.datasets!=['gsm8k']:
        parser.error('This Phi phase is GSM8K evaluation + three-layer collection only.')
    if args.mode=='summary':
        for name in args.datasets:
            folder=args.output/args.model/name
            if not (folder/'run_config.json').exists():
                print(f'{args.model}/{name}: not prepared yet.',flush=True)
                continue
            plan=json.loads((folder/'run_config.json').read_text())
            plans={e:json.loads((folder/e/'run_config.json').read_text()) for e in EMOTIONS}
            report(folder,plan,plans)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/(args.model+'-runner.lock')).open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        registry=Registry(audio_root(args.audio_root))
        cases=[prepare(name,registry,args.output,args.model) for name in args.datasets]
        for folder,plan,plans in cases:
            if args.mode=='prepare':
                winners.score_rows(dict(target=plan),[],folder/'scoring-preflight')
        if args.mode=='prepare':
            print('PREPARED. No GPU generation or classification.',flush=True)
            return
        with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
            print('Waiting for shared GPU lock and idle GPU.',flush=True)
            fcntl.flock(gpu,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                time.sleep(30)
            backend=DiffaBackend() if args.model=='diffa' else PhiBackend()
            for folder,plan,plans in cases:
                from importlib.metadata import version
                from diffa_model_receipts import model_revisions
                config=backend.model.config if args.model=='diffa' else backend.native.model.config
                winners.atomic_text(folder/'runtime.json',json.dumps(dict(
                    runner_sha256=winners.digest(__file__),block_collector_sha256=winners.digest(blocks.__file__),
                    model_revisions=model_revisions() if args.model=='diffa' else plan['recipe'],
                    packages={k:version(k) for k in ('torch','transformers')},
                    model_config=config.to_dict()),indent=2,default=str)+'\n')
                print(f'STAGE {args.model}/{plan["name"]}: all four natural deliveries and simultaneous agreed fitting.',flush=True)
                generate_dataset(folder,plan,plans,backend)
                if args.model=='diffa':
                    natural_report(folder,plan)
                    steer_dataset(folder,plan,plans,backend)
                else:
                    command=[str(Path('.venv-diffa/bin/python').resolve()),'-u',__file__,'--mode','summary',
                             '--model','phi','--datasets',plan['name'],'--output',str(args.output.resolve())]
                    subprocess.run(command,check=True)
                print(f'DONE {args.model}/{plan["name"]}',flush=True)


if __name__=='__main__':
    main()
