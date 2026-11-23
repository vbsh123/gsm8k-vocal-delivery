"""DIFFA happy: historical collection/test IDs, native delivery, strengths and transfers."""
import argparse
import fcntl
import json
import math
import shutil
import os
from pathlib import Path
import statistics
import subprocess
import time

import diffa_gsm_emotion as emotion
import winning_diffa_sweeps as winners
from extract_audio_vectors import average_pair, save_tensor, GROUPS
from steer import read_rows

VOICE='en-US-JennyNeural'
DATASETS=('gsm8k','svamp','multiarith','bbh3','bbh5','ifeval')
STAGES=('primary','strengths','transfer')
PROTOCOL='diffa-happy-winners-v1'


def audio_inventory(root,styles=('neutral','happy')):
    """Only Jenny/Azure; never mix Ryan, model predictions or classifier labels."""
    found={}
    for path in sorted(root.rglob('plan.jsonl')):
        for row in read_rows(path):
            if not isinstance(row,dict):continue
            if not all(isinstance(row.get(k),str) and row[k] for k in ('question_id','question','audio_path')):continue
            if row.get('provider')!='azure' or row.get('voice')!=VOICE or row.get('style') not in styles:continue
            if 'question' not in row or not (path.parent/row['audio_path']).is_file():continue
            found.setdefault((row['question_id'],row['style']),[]).append((row,path.parent))
    return found


def normal_answer(value):
    return '' if value is None else str(value).rsplit('####',1)[-1].strip()


def make_record(row,folder):
    result=emotion.recording(dict(row,answer=normal_answer(row.get('answer'))),folder)
    result['reference_answer']=normal_answer(row.get('answer'))
    return result


def historical_collection_ids(target):
    ids=target['extraction_ids']
    count=target['vector_info'].get('pairs',len(ids))
    if count==len(ids):return ids
    import torch
    state=torch.load(target['vector'],map_location='cpu',weights_only=True)
    actual=[p['question_id'] for p in state.get('pairs',[])]
    if len(actual)!=count or len(set(actual))!=count or not set(actual)<=set(ids):
        raise ValueError('Historical exclusions differ from contributing pairs; need the saved actual pair IDs.')
    return actual


def task_plan(name,entry,root,inventory,require_existing_audio=False):
    folder=root/name;folder.mkdir(parents=True,exist_ok=True)
    frozen=folder/'run_config.json'
    if frozen.exists():
        plan=json.loads(frozen.read_text())
        if plan['protocol']!=PROTOCOL or plan['source_run']!=str(Path(entry['run']).absolute()):raise ValueError('Frozen task differs.')
        if (winners.digest(plan['target']['config_file'])!=plan['source_config_sha256'] or
                winners.digest(plan['target']['vector'])!=plan['source_vector_sha256']):raise ValueError('Original source receipts changed.')
        for row in [r for p in plan['pairs'] for r in p.values() if isinstance(r,dict)]+plan['recordings']:
            if winners.digest(row['audio_file'])!=row['audio_sha256']:raise ValueError('Frozen audio missing/changed.')
        return plan
    target=winners.load_dataset(name,entry)
    config=json.loads(Path(target['config_file']).read_text())
    train=historical_collection_ids(target);test=[r['question_id'] for r in target['records']]
    if not train or len(set(train))!=len(train) or set(train)&set(test):raise ValueError('Missing/overlapping historical collection IDs.')
    # Preserve raw angry-vector collection IDs, including historical GSM classifier selection.
    collection_neutral={p['question_id']:p['neutral'] for p in config.get('extraction',[])}
    bank={}; neutral={}; all_ids=train+test
    tests={r['question_id']:r for r in target['records']}
    for qid in all_ids:
        options=inventory.get((qid,'neutral'),[])
        expected=tests.get(qid,collection_neutral.get(qid))
        if expected:
            sha=expected['audio_sha256']
            options=[x for x in options if winners.digest(x[1]/x[0]['audio_path'])==sha]
        if not options:raise FileNotFoundError(f'{name}: restore original neutral audio/plan for {qid}; neutral TTS is not requested.')
        row,source=options[0]
        q=target['bank'].get(qid)
        if q and q.get('question',q.get('prompt',row['question']))!=row['question']:raise ValueError('Frozen question/audio text differs.')
        bank[qid]=dict(question_id=qid,question=row['question'],answer=normal_answer(row.get('answer')))
        if q and name!='ifeval' and normal_answer(q['answer'])!=bank[qid]['answer']:raise ValueError('Frozen reference differs.')
        neutral[qid]=make_record(row,source)
    happy={}
    for qid in all_ids:
        for row,source in inventory.get((qid,'happy'),[]):
            if row['question']==bank[qid]['question'] and normal_answer(row.get('answer'))==bank[qid]['answer']:
                happy[qid]=make_record(row,source);break
    destination=folder/'audio-happy'
    missing=[bank[qid] for qid in all_ids if qid not in happy]
    if missing:
        if require_existing_audio:
            raise FileNotFoundError(f'{name}: {len(missing)} happy clips missing; existing-audio-only queue makes no TTS requests.')
        questions=read_rows(destination/'questions.jsonl') or missing
        print(f'{name}: generating {len(missing)} missing happy clips only.',flush=True)
        from run_reasoning_transfer import generate_audio
        generate_audio(destination,questions,['happy'])
        for row in read_rows(destination/'plan.jsonl'):
            qid=row['question_id']
            if qid in bank and row['question']==bank[qid]['question'] and row['style']=='happy':
                happy[qid]=make_record(row,destination)
    if set(happy)!=set(all_ids):raise ValueError('Incomplete happy audio.')
    if name=='ifeval':
        long=[qid for qid in all_ids if happy[qid]['seconds']>30]
        if long:print(f'IFEval: {len(long)} happy clips exceed30s; preserve all frozen IDs and use full chunked audio. This differs from historical four-style length eligibility.',flush=True)
    plan=dict(protocol=PROTOCOL,name=name,emotion='happy',source_run=target['run'],target=target,
        source_config_sha256=target['config_sha256'],source_vector_sha256=target['vector_sha256'],
        collection_ids=train,test_ids=test,bank=bank,
        pairs=[dict(question_id=q,neutral=neutral[q],emotion=happy[q]) for q in train],
        recordings=[dict((neutral if style=='neutral' else happy)[q],condition=style) for q in test for style in ('neutral','happy')],
        prompt=target['prompt'],collection_prompt=target['vector_info']['collection_prompt'],
        seed=42,max_new_tokens=target['max_new_tokens'],steps=128,block_length=32,decoder='fast_dllm',temperature=0.,
        averaging='Matched forwards within each pair; token-position means; equal per-question weight.',
        selection='Exact historical angry-vector collection IDs and held-out target IDs. No new outcome/classifier filtering.',
        note='Historical collection selection is inherited, including GSM classifier-selected IDs. Happy labels are not filtered. '
             'Original collection and target prompts/budgets preserved separately; full audio, native<=30s/chunked Whisper above30s.')
    winners.freeze(frozen,plan)
    print(f'{name}: {len(train)} collection pairs / {len(test)} test questions; budget{plan["max_new_tokens"]}.',flush=True)
    return plan


def generate(record,plan,prompt,model,tokenizer,features):
    import soundfile as sf
    from extract_vectors import gen_chunked
    if winners.digest(record['audio_file'])!=record['audio_sha256']:raise ValueError('Audio changed.')
    audio,sr=sf.read(record['audio_file'],dtype='float32');winners.seed()
    return gen_chunked(audio,sr,model,tokenizer,features,prompt,plan['max_new_tokens'],30)


def collect(plan,folder,model,tokenizer,features):
    import torch
    import diffa_stream_collection as running
    config_hash=winners.digest(folder/'run_config.json')
    state=running.load(plan,folder,config_hash)
    for i in range(len(state['completed_ids']),len(plan['pairs'])):
        pair=plan['pairs'][i];path=folder/'pairs'/f'{i:05d}.pt'
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['run_config_sha256']!=config_hash or saved['question_id']!=pair['question_id']:raise ValueError('Pair cache differs.')
        else:
            import src.fast_dllm.diffa_generate as decoder
            print(f'{plan["name"]} collect [{i+1}/{len(plan["pairs"])}] {pair["question_id"]}; running aggregate.',flush=True)
            traces=[]
            for style in ('neutral','emotion'):
                record=pair[style]
                traces.append(emotion.capture(model,lambda:generate(record,plan,plan['collection_prompt'],model,tokenizer,features),decoder,plan['max_new_tokens']))
            nr,ne,n=traces[0];hr,he,h=traces[1]
            nm,hm,delta,matched=average_pair(n,h,'matched')
            saved=dict(protocol=plan.get('protocol',PROTOCOL),run_config_sha256=config_hash,question_id=pair['question_id'],
                neutral=nm,emotion=hm,difference=delta,neutral_response=nr,emotion_response=hr,
                neutral_forwards=n.shape[1],emotion_forwards=h.shape[1],matched_forwards=matched,
                neutral_encoding=ne,emotion_encoding=he)
            del traces,n,h,nm,hm,delta
        running.append(state,saved)
        save_tensor(torch,state,folder/'collection-stream.pt')
        del saved
    running.finalize(plan,folder,state)
    print(f'{plan["name"]}: all-layer vectors, drift and raw collection answers saved; no large new pair caches.',flush=True)


def reuse_delivery(plan,folder):
    target=plan['target'];tasks={(r['question_id'],r['voice'],r['condition']):r for r in plan['recordings']}
    done={(r['question_id'],r['voice'],r['condition']):r for r in read_rows(folder/'results.jsonl',repair=True)}
    sources=[Path(target['run']),*map(Path,target['reuse_runs'])]
    if plan['name']=='gsm8k':sources.append(Path('outputs/diffa-gsm8k-happy/evaluations/L31-answer-a1'))
    if plan['name']=='ifeval':sources.append(Path('outputs/spoken-ifeval-diffa2'))
    if plan['name']=='svamp':sources.append(Path('outputs/svamp500-diffa2-hashes'))
    # Also reuse compatible historical happy deliveries outside the winning steering folders.
    for path in sorted({*Path('outputs').rglob('run_config.json'),*Path('outputs').rglob('split.json')}):
        if not (path.parent/'results.jsonl').exists():continue
        try:config=json.loads(path.read_text())
        except json.JSONDecodeError:continue
        if (config.get('decoder')=='fast_dllm' and config.get('prompt')==target['prompt'] and
                config.get('max_new_tokens')==target['max_new_tokens']):sources.append(path.parent)
    for source in dict.fromkeys(sources):
        try:config=json.loads(winners.config_file(source).read_text())
        except FileNotFoundError:continue
        for row in read_rows(source/'results.jsonl'):
            label=row.get('condition',row.get('style'));key=row['question_id'],row['voice'],label
            if key in done or key not in tasks or row.get('interventions'):continue
            record=tasks[key]
            if not winners.reusable(row,config,record,target):continue
            saved=dict(record,response=row['response'],strength=0.,prompt=plan['prompt'],seed=42,
                max_new_tokens=plan['max_new_tokens'],steps=128,block_length=32,temperature=0.,decoder='fast_dllm',
                interventions=[],reused_from=str(source),response_tokens=row.get('response_tokens'),audio_encoding=row.get('audio_encoding'))
            with (folder/'results.jsonl').open('a') as stream:stream.write(json.dumps(saved)+'\n')
            done[key]=saved
    missing=[r['question_id'] for r in plan['recordings'] if r['condition']=='neutral' and (r['question_id'],r['voice'],'neutral') not in done]
    if missing:raise ValueError(f'{plan["name"]}: {len(missing)} compatible neutral answers missing; restore baseline, no neutral evaluation scheduled.')
    print(f'{plan["name"]}: {len(done)} held-out answers reused; no neutral generations.',flush=True)


def deliver(plan,folder,model,tokenizer,features):
    label=plan['emotion']
    done={(r['question_id'],r['voice'],r['condition']) for r in read_rows(folder/'results.jsonl',repair=True)}
    pending=[r for r in plan['recordings'] if r['condition']==label and (r['question_id'],r['voice'],label) not in done]
    with (folder/'results.jsonl').open('a') as stream:
        for i,record in enumerate(pending,1):
            print(f'{plan["name"]} {label} [{i}/{len(pending)}] {record["question_id"]}',flush=True)
            response,encoding=generate(record,plan,plan['prompt'],model,tokenizer,features)
            stream.write(json.dumps(dict(record,response=response,strength=0.,prompt=plan['prompt'],seed=42,
                max_new_tokens=plan['max_new_tokens'],steps=128,block_length=32,temperature=0.,decoder='fast_dllm',
                interventions=[],audio_encoding=encoding,response_tokens=len(tokenizer.encode(response,add_special_tokens=False))))+'\n');stream.flush()


def descriptor(plan,folder):
    target=dict(plan['target'],emotion=plan['emotion'],vector=str((folder/'style-vectors.pt').absolute()),
        vector_sha256=winners.digest(folder/'style-vectors.pt'),extraction_ids=plan['collection_ids'])
    _,target['vector_info']=winners.read_vector(target['vector'],plan['emotion'])
    return target


def steering_plan(source,target,strengths):
    if set(source['extraction_ids']) & {r['question_id'] for r in target['records']}:raise ValueError('Transfer collection overlaps target test.')
    return dict(protocol=PROTOCOL,kind='happy-steering',source=source,target=target,strengths=list(strengths),
        layer=31,group='answer',schedule='always',mode='fixed')


def reuse_neutral(delivery,output):
    done={(r['question_id'],r['voice'],r['condition']) for r in read_rows(output/'results.jsonl',repair=True)}
    with (output/'results.jsonl').open('a') as stream:
        for row in read_rows(delivery/'results.jsonl'):
            key=row['question_id'],row['voice'],'neutral'
            if row['condition']!='neutral' or key in done:continue
            stream.write(json.dumps(dict(row,condition='neutral',strength=0.,reused_from=str(delivery)))+'\n');done.add(key)


def report_delivery(plan,folder,tokenizer=None,*,rows=None,cohort='held-out',expected_ids=None):
    from rescore_audio_steering import paired
    label=plan['emotion']
    rows=read_rows(folder/'results.jsonl') if rows is None else rows
    expected_ids=plan['test_ids'] if expected_ids is None else expected_ids
    available={c:{(r['question_id'],r['voice']) for r in rows if r['condition']==c} for c in ('neutral',label)}
    if not available['neutral'] & available[label]:
        print(f'{plan["name"]} {plan["emotion"]} vs neutral: 0/{len(expected_ids)} matched {cohort} questions; neutral={len(available["neutral"])} {label}={len(available[label])} saved.',flush=True)
        return
    scoring=dict(target=plan['target'])
    scored,provenance=winners.score_rows(scoring,rows,folder)
    groups={c:{(r['question_id'],r['voice']):r for r in scored if r['condition']==c} for c in ('neutral',label)}
    keys=sorted(groups['neutral'].keys()&groups[label].keys());results=[];lengths={}
    if tokenizer is None and any(groups[c][k].get('response_tokens') is None for c in groups for k in keys):
        from transformers import AutoTokenizer
        from diffa_model_receipts import model_revisions
        tokenizer=AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct',
            revision=model_revisions().get('GSAI-ML/LLaDA-8B-Instruct'),local_files_only=True)
    for c,g in groups.items():
        counts=[r.get('response_tokens') if r.get('response_tokens') is not None else len(tokenizer.encode(r['response'],add_special_tokens=False)) for r in (g[k] for k in keys)]
        if counts:lengths[c]=dict(n=len(counts),mean_tokens=statistics.mean(counts),median_tokens=statistics.median(counts))
    print(f'{plan["name"]} {plan["emotion"]} vs neutral: {len(keys)}/{len(expected_ids)} matched {cohort} questions.',flush=True)
    for c,info in lengths.items():print(c,'mean/median tokens:',round(info['mean_tokens'],1),info['median_tokens'],flush=True)
    instruction_totals=[]
    if plan['name']=='ifeval':
        for c,g in groups.items():
            for mode in ('strict','loose'):
                bits=[b for k in keys for b in g[k]['official_scores'][mode]['instructions']]
                if bits:instruction_totals.append(dict(condition=c,mode=mode,instructions=len(bits),correct=sum(bits),accuracy=sum(bits)/len(bits)))
    if keys:
        for metric in groups['neutral'][keys[0]]['metrics']:
            for c,g in groups.items():
                correct=sum(g[k]['metrics'][metric] for k in keys);print(f'{metric} / {c}: {correct}/{len(keys)} ({correct/len(keys):.1%})',flush=True)
            result=paired([(groups['neutral'][k]['metrics'][metric],groups[label][k]['metrics'][metric]) for k in keys])
            results.append(dict(metric=metric,**result));print(f'{metric}: gain={result["gain_pp"]:+.2f}; p={result["mcnemar_exact_p"]:.4g}; CI={result["ci95_gain_pp"]}',flush=True)
    winners.atomic_text(folder/'summary.json',json.dumps(dict(paired=results,lengths=lengths,instruction_totals=instruction_totals,scorer=provenance,cohort=cohort,expected_questions=len(expected_ids),matched_questions=len(keys),prompt=plan['prompt'],note='Exploratory unadjusted paired tests; '+('collection responses used for vector fitting, not held-out steering evidence.' if cohort=='collection' else 'same historical held-out cohort.')),indent=2)+'\n')
    winners.atomic_text(folder/'rescored.jsonl',''.join(json.dumps(r)+'\n' for r in scored))


def report_collection(plan,folder,tokenizer=None):
    from diffa_collection_report import collection_rows,collection_bank
    rows=collection_rows(plan,folder)
    if not rows:
        print(f'{plan["name"]} {plan["emotion"]} vs neutral: 0/{len(plan["collection_ids"])} matched collection questions saved yet.',flush=True)
        return
    bank=collection_bank(plan)
    target=dict(plan['target'],bank={q:bank[q] for q in plan['collection_ids']})
    scoped=dict(plan,target=target,prompt=plan['collection_prompt'])
    output=folder/'collection-report';output.mkdir(exist_ok=True)
    winners.atomic_text(output/'results.jsonl',''.join(json.dumps(r)+'\n' for r in rows))
    winners.atomic_text(output/'scoring-bank.json',json.dumps(dict(task=plan['name'],cohort='collection',
        prompt=plan['collection_prompt'],bank=target['bank']),indent=2)+'\n')
    print('Collection comparison: responses used for vector fitting; reported separately from held-out results.',flush=True)
    if plan['collection_prompt']!=plan['prompt']:print('Collection uses its original collection prompt, which differs from the held-out evaluation prompt.',flush=True)
    if plan['name'] in ('gsm8k','svamp','multiarith') and '####' not in plan['collection_prompt']:
        print('Collection prompt does not request ####; upstream strict/format metrics are extraction diagnostics, not compliance with that collection prompt.',flush=True)
    report_delivery(scoped,output,tokenizer,rows=rows,cohort='collection',expected_ids=plan['collection_ids'])


def report_steering(folder,tokenizer=None):
    winners.report_job(folder)
    rows=read_rows(folder/'results.jsonl');lengths={}
    groups={}
    for row in rows:groups.setdefault(row['condition'],{})[row['question_id'],row['voice']]=row
    common=set.intersection(*(set(g) for g in groups.values())) if len(groups)>1 else set()
    matched=[row for row in rows if (row['question_id'],row['voice']) in common]
    note=None
    if tokenizer is None and any(row.get('response_tokens') is None for row in matched):
        try:
            from transformers import AutoTokenizer
            from diffa_model_receipts import model_revisions
            revision=model_revisions().get('GSAI-ML/LLaDA-8B-Instruct')
            tokenizer=AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct',revision=revision,local_files_only=True)
        except (OSError,ImportError,ValueError) as error:
            note=f'Token lengths unavailable: saved counts missing and local tokenizer unavailable ({type(error).__name__}). Accuracy scores remain valid.'
    if note is None:
        for row in matched:
            count=row.get('response_tokens')
            if count is None:count=len(tokenizer.encode(row['response'],add_special_tokens=False))
            lengths.setdefault(row['condition'],[]).append(count)
    if not matched:
        note='No matched steered answers saved yet; only baseline scores are available.'
    path=folder/'summary.json'
    if path.exists():
        summary=json.loads(path.read_text());summary['lengths']={k:dict(n=len(v),mean_tokens=statistics.mean(v),median_tokens=statistics.median(v)) for k,v in lengths.items()}
        if note:summary['lengths_note']=note
        else:summary.pop('lengths_note',None)
        winners.atomic_text(path,json.dumps(summary,indent=2)+'\n')
        for c,info in summary['lengths'].items():print(c,'matched mean/median tokens:',round(info['mean_tokens'],1),info['median_tokens'],flush=True)
    if note:print(note,flush=True)


def summarize(root,names):
    for name in names:
        path=root/name/'run_config.json'
        if not path.exists():print(name,'not prepared.');continue
        plan=json.loads(path.read_text());report_collection(plan,root/name)
        report_delivery(plan,root/name/'delivery')
        for folder in sorted((root/name).glob('strength-*')):report_steering(folder)
    for folder in sorted((root/'transfer').glob('*')):
        if (folder/'run_config.json').exists():report_steering(folder)


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('run','prepare','summary'),default='run')
    p.add_argument('--datasets',choices=DATASETS,nargs='+',default=list(DATASETS))
    p.add_argument('--stages',choices=STAGES,nargs='+',default=['primary'])
    p.add_argument('--strengths',type=float,nargs='+',default=[1.])
    p.add_argument('--catalog',type=Path,default=Path('winning_diffa_catalog.json'))
    p.add_argument('--output',type=Path,default=Path('outputs/diffa-happy-winners'))
    p.add_argument('--require-existing-audio',action='store_true',help='Fail rather than generating any missing TTS audio.')
    args=p.parse_args(argv)
    if not args.strengths or not all(math.isfinite(x) for x in args.strengths):p.error('Strengths must be finite.')
    return args


def main():
    args=arguments()
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    if args.mode=='summary':summarize(root,args.datasets);return
    with (root/'queue.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        entries=json.loads(args.catalog.read_text())
        # Validate the six original sources before spending on TTS or GPU work.
        for name in args.datasets:winners.load_dataset(name,entries[name])
        inventory=audio_inventory(Path('outputs'));plans={}
        for name in args.datasets:
            plan=task_plan(name,entries[name],root,inventory,require_existing_audio=args.require_existing_audio);plans[name]=plan
            delivery=root/name/'delivery';delivery.mkdir(exist_ok=True)
            winners.freeze(delivery/'run_config.json',dict(protocol=PROTOCOL,target=plan['target'],conditions=['neutral','happy']))
            reuse_delivery(plan,delivery)
            from diffa_collection_report import collection_bank
            collection_bank(plan)  # Resolve collection scoring metadata before GPU work.
            winners.score_rows(dict(target=plan['target']),[],delivery)  # Fail on missing evaluator before GPU work.
        if args.mode=='prepare':return
        # Running sums are constant-size; no reservation for every pair activation tensor.
        reserve=512*2**20+sum(32*len(GROUPS)*plan['target']['vector_info']['width']*8*4 for plan in plans.values())
        free=shutil.disk_usage(root).free
        if free < reserve:
            raise RuntimeError(f'Need approximately{reserve/2**30:.2f} GiB free for aggregate checkpoints/results; have{free/2**30:.2f}. No files deleted.')
        print(f'Running-aggregate collection: {free/2**30:.2f} GiB free; {reserve/2**30:.2f} GiB reserve. No full pair-cache requirement.',flush=True)
        winners.freeze(root/'queue_manifest.json',dict(protocol=PROTOCOL,datasets=args.datasets,stages=args.stages,
            strengths=args.strengths,layer=31,group='answer',transfer_strength=1.,
            order='Happy collection, delivery comparison and own x1 steering only.' if args.stages==['primary'] else
                'Explicitly selected stages: primary first, then requested strengths/transfers.',
            note='Exploratory. Same collection/test IDs and historical prompts/budgets; no held-out neutral inference.'))
        with Path('outputs/jenny-task-vectors.lock').open('a') as lock:
            print('Waiting for shared DIFFA GPU lock.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
            cache=Path(os.environ.get('HF_HOME','/workspace/.hf_home')).absolute()
            os.environ.update(HF_HOME=str(cache),HF_HUB_CACHE=str(cache/'hub'),HUGGINGFACE_HUB_CACHE=str(cache/'hub'));os.environ.pop('TRANSFORMERS_CACHE',None)
            from extract_vectors import load_model
            model,tokenizer,features=load_model()
            winners.atomic_text(root/'runtime.json',json.dumps(dict(runner_sha256=winners.digest(__file__),
                hooks_sha256=winners.digest(emotion.__file__),stream_collector_sha256=winners.digest(Path(__file__).with_name('diffa_stream_collection.py')),generation_sha256=winners.digest(Path(__file__).with_name('extract_vectors.py')),
                model_config=model.config.to_dict()),indent=2,default=str)+'\n')
            descriptors={}
            for name,plan in plans.items():
                folder=root/name
                collect(plan,folder,model,tokenizer,features);report_collection(plan,folder,tokenizer)
                descriptors[name]=descriptor(plan,folder)
                if 'primary' in args.stages:
                    deliver(plan,folder/'delivery',model,tokenizer,features);report_delivery(plan,folder/'delivery',tokenizer)
                    job=steering_plan(descriptors[name],descriptors[name],[0.,1.]);out=folder/'strength-1';out.mkdir(exist_ok=True)
                    winners.freeze(out/'run_config.json',job);reuse_neutral(folder/'delivery',out)
                    winners.generate_job(job,out,model,tokenizer,features);report_steering(out,tokenizer)
            if 'strengths' in args.stages:
                for name,target in descriptors.items():
                    for strength in dict.fromkeys(args.strengths):
                        if not strength:continue
                        out=root/name/f'strength-{strength:g}';out.mkdir(exist_ok=True)
                        job=steering_plan(target,target,[0.,strength]);winners.freeze(out/'run_config.json',job)
                        reuse_neutral(root/name/'delivery',out);winners.generate_job(job,out,model,tokenizer,features);report_steering(out,tokenizer)
            if 'transfer' in args.stages:
                for source_name,source in descriptors.items():
                    for target_name,target in descriptors.items():
                        if source_name==target_name:continue
                        out=root/'transfer'/f'{target_name}__from__{source_name}';out.mkdir(parents=True,exist_ok=True)
                        job=steering_plan(source,target,[0.,1.]);winners.freeze(out/'run_config.json',job)
                        reuse_neutral(root/target_name/'delivery',out);winners.generate_job(job,out,model,tokenizer,features);report_steering(out,tokenizer)

if __name__=='__main__':main()
