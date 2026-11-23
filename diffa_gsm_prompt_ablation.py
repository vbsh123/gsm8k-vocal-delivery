"""Fixed original/shared3 GSM angry+happy vectors under concise and older prompts; no recollection."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import statistics
import subprocess
import time

import winning_diffa_sweeps as winners
from steer import read_rows

PROTOCOL='diffa-gsm-original-shared3-prompt-vector-ablation-v1'
COHORTS=('original','shared3')
CONDITIONS=('neutral','angry-vector','happy-vector')


def descriptor(plan,folder,emotion):
    if emotion=='angry' and plan['emotion']=='happy':
        result=deepcopy(plan['target'])
        result['extraction_ids']=plan['collection_ids']
    else:
        import diffa_happy_nightly as task
        result=task.descriptor(plan,folder)
    result['emotion']=emotion
    path=Path(result['vector'])
    _,info=winners.read_vector(path,emotion)
    if info['pairs']!=len(plan['collection_ids']):
        raise ValueError(f'Vector is not complete or has a different contributing cohort: {path}')
    result.update(vector=str(path.absolute()),vector_sha256=winners.digest(path),vector_info=info)
    return result


def prepare(cohort,root,output):
    from evaluate_gsm_candidates import ORIGINAL
    if cohort=='original':
        path=root/'run_config.json'
        plans={e:json.loads(path.read_text()) for e in ('angry','happy')}
        sources={e:root for e in plans}
        expected=(425,718)
    else:
        plans={e:json.loads((root/e/'run_config.json').read_text()) for e in ('angry','happy')}
        sources={e:root/e for e in plans}
        expected=(609,660)
    template=plans['happy']
    train,test=template['collection_ids'],template['test_ids']
    if (len(train),len(test))!=expected or len(set(train+test))!=len(train+test):
        raise ValueError(f'Need the frozen {cohort} GSM cohort {expected}, without resplitting.')
    target=deepcopy(template['target'])
    if [r['question_id'] for r in target['records']]!=test:
        raise ValueError('Target record order differs from frozen held-out IDs.')
    vectors={e:descriptor(plans[e],sources[e],e) for e in plans}
    for e in plans:
        if plans[e]['collection_ids']!=train or plans[e]['test_ids']!=test:
            raise ValueError('Angry and happy fitting/evaluation IDs differ.')
        if vectors[e]['records']!=target['records']:
            raise ValueError('Angry and happy evaluation recordings differ.')
        if set(vectors[e]['extraction_ids'])&set(test):
            raise ValueError('Vector fitting overlaps evaluation questions.')
        for field in ('seed','max_new_tokens','steps','block_length','temperature','decoder'):
            if vectors[e][field]!=target[field]:raise ValueError(f'Emotion generation settings differ: {field}')
    concise=target['prompt']
    older=concise.replace('Give concise step-by-step reasoning with only the necessary calculations.',"Let's think step by step.")
    if older!=ORIGINAL or older==concise:
        raise ValueError('Unexpected GSM prompt; change only the reasoning instruction.')
    for record in target['records']:
        if winners.digest(record['audio_file'])!=record['audio_sha256']:
            raise ValueError(f'Target neutral recording is absent/changed: {record["question_id"]}')
    references=[]
    if cohort=='original':
        references=[str(root/'strength-1'),str(root/'delivery'),'outputs/winning-strengths/gsm8k',target['run']]
    else:
        references=[str(root/e/'primary') for e in ('angry','happy')]
    plan=dict(protocol=PROTOCOL,cohort=cohort,target=target,vectors=vectors,collection_ids=train,test_ids=test,
              prompts=dict(concise=concise,older=older),conditions=list(CONDITIONS),strength=1.,layer=31,group='answer',
              references=references,source_configs={e:dict(path=str((sources[e]/'run_config.json').absolute()),
                    sha256=winners.digest(sources[e]/'run_config.json')) for e in plans},
              note='Fixed original/shared3 vectors and their own frozen held-out cohorts. Same neutral audio for both emotions. '
                   'No natural angry/happy delivery, TTS, e2v inference, vector fitting, normalization or cohort selection. '
                   'Only prompt instruction changes; compare each vector against its own prompt baseline.')
    output.mkdir(parents=True,exist_ok=True)
    winners.freeze(output/'run_config.json',plan)
    print(f'{cohort}: {len(train)} frozen fitting IDs; {len(test)} evaluation IDs × 2 prompts × 3 conditions; +1, no recollection.',flush=True)
    return plan


def key(row):return row['question_id'],row['voice'],row['prompt_key'],row['condition']


def validate(row,plan,config_hash):
    records={(r['question_id'],r['voice']):r for r in plan['target']['records']}
    q,v,prompt,condition=key(row)
    if (q,v) not in records or prompt not in plan['prompts'] or condition not in CONDITIONS:
        raise ValueError('Answer outside frozen prompt/cohort/condition.')
    emotion=condition.removesuffix('-vector') if condition!='neutral' else None
    vector_hash=plan['vectors'][emotion]['vector_sha256'] if emotion else None
    expected=dict(run_config_sha256=config_hash,prompt=plan['prompts'][prompt],
                  audio_sha256=records[q,v]['audio_sha256'],reference_answer=records[q,v]['reference_answer'],
                  strength=1. if emotion else 0.,vector_sha256=vector_hash,
                  interventions=[dict(layer=31,group='answer',strength=1.)] if emotion else [])
    expected.update({f:plan['target'][f] for f in ('seed','max_new_tokens','steps','block_length','temperature','decoder')})
    if any(row.get(k)!=value for k,value in expected.items()):
        raise ValueError('Saved prompt ablation answer provenance differs from frozen settings.')


def answers(plan,folder,repair=False):
    result={}
    config_hash=winners.digest(folder/'run_config.json')
    for row in read_rows(folder/'results.jsonl',repair=repair):
        validate(row,plan,config_hash)
        if key(row) in result:raise ValueError('Duplicate ablation answer.')
        result[key(row)]=row
    return result


def copied(record,row,plan,folder,prompt,condition,source):
    emotion=condition.removesuffix('-vector') if condition!='neutral' else None
    result=dict(record,response=row['response'],response_tokens=row.get('response_tokens'),
                audio_encoding=row.get('audio_encoding'),prompt_key=prompt,condition=condition,
                prompt=plan['prompts'][prompt],run_config_sha256=winners.digest(folder/'run_config.json'),
                strength=1. if emotion else 0.,vector_sha256=plan['vectors'][emotion]['vector_sha256'] if emotion else None,
                interventions=[dict(layer=31,group='answer',strength=1.)] if emotion else [],reused_from=str(source))
    result.update({f:plan['target'][f] for f in ('seed','max_new_tokens','steps','block_length','temperature','decoder')})
    return result


def reuse(plan,folder,other_folders=()):
    done=answers(plan,folder,repair=True)
    records={(r['question_id'],r['voice']):r for r in plan['target']['records']}
    with (folder/'results.jsonl').open('a') as stream:
        for root in map(Path,plan['references']):
            if not (root/'results.jsonl').exists():continue
            try:config=json.loads(winners.config_file(root).read_text())
            except FileNotFoundError:continue
            for row in read_rows(root/'results.jsonl'):
                record=records.get((row['question_id'],row['voice']))
                if record is None:continue
                strength=float(row.get('strength') or 0.)
                if row.get('condition',row.get('style'))=='neutral' and not row.get('interventions') and strength==0.:
                    condition='neutral'
                elif strength==1. and row.get('interventions')==[dict(layer=31,group='answer',strength=1.)]:
                    source=config.get('source',{})
                    source=source if isinstance(source,dict) else {}
                    if row.get('vector_sha256') and source.get('vector_sha256') and row['vector_sha256']!=source['vector_sha256']:
                        raise ValueError('Cached steering row disagrees with its source-vector receipt.')
                    vector_hash=row.get('vector_sha256') or source.get('vector_sha256')
                    emotion=next((e for e in ('angry','happy') if plan['vectors'][e]['vector_sha256']==vector_hash),None)
                    if emotion is None:continue
                    condition=emotion+'-vector'
                else:continue
                for prompt,text in plan['prompts'].items():
                    target=dict(plan['target'],prompt=text)
                    if not winners.reusable(row,config,record,target):continue
                    result=copied(record,row,plan,folder,prompt,condition,root)
                    if key(result) in done:continue
                    stream.write(json.dumps(result)+'\n');done[key(result)]=result
        # Reuse prompt-matched neutral baselines across overlapping protocol cohorts.
        for other in other_folders:
            other=Path(other)
            if other==folder or not (other/'run_config.json').exists():continue
            other_plan=json.loads((other/'run_config.json').read_text())
            for row in answers(other_plan,other).values():
                record=records.get((row['question_id'],row['voice']))
                if record is None or row['condition']!='neutral':continue
                prompt=row['prompt_key']
                if prompt not in plan['prompts'] or row['prompt']!=plan['prompts'][prompt]:continue
                if not winners.reusable(row,other_plan,record,dict(plan['target'],prompt=row['prompt'])):continue
                result=copied(record,row,plan,folder,prompt,'neutral',other)
                if key(result) not in done:
                    stream.write(json.dumps(result)+'\n');done[key(result)]=result
    return done


def pending(plan,done):
    return [(record,prompt,condition) for prompt in plan['prompts'] for condition in CONDITIONS
            for record in plan['target']['records'] if (record['question_id'],record['voice'],prompt,condition) not in done]


def generate(plan,folder,model,tokenizer,features):
    import torch
    import soundfile as sf
    from extract_vectors import gen_chunked
    from steer import install_interventions
    vectors={e:winners.read_vector(d['vector'],e)[0].to('cuda',dtype=torch.bfloat16) for e,d in plan['vectors'].items()}
    width=model.llm_model.get_input_embeddings().weight.shape[1]
    if any(v.shape!=(width,) for v in vectors.values()):raise ValueError('Fixed vector width differs from DIFFA hidden dimension.')
    layers=model.llm_model.model.transformer.blocks
    todo=pending(plan,answers(plan,folder,repair=True))
    with (folder/'results.jsonl').open('a') as stream:
        for index,(record,prompt,condition) in enumerate(todo,1):
            if winners.digest(record['audio_file'])!=record['audio_sha256']:raise ValueError('Evaluation audio changed.')
            audio,sr=sf.read(record['audio_file'],dtype='float32')
            winners.seed()
            emotion=condition.removesuffix('-vector') if condition!='neutral' else None
            hooks=install_interventions(layers,{(31,'answer'):vectors[emotion]},[dict(layer=31,group='answer',strength=1.)],{},plan['target']['max_new_tokens']) if emotion else nullcontext()
            print(f'{plan["cohort"]}/{prompt}/{condition} [{index}/{len(todo)}] {record["question_id"]}',flush=True)
            with hooks:
                response,encoding=gen_chunked(audio,sr,model,tokenizer,features,plan['prompts'][prompt],plan['target']['max_new_tokens'],30)
            row=copied(record,dict(response=response,response_tokens=len(tokenizer.encode(response,add_special_tokens=False)),audio_encoding=encoding),
                       plan,folder,prompt,condition,'generated-here')
            row['reused_from']=None
            stream.write(json.dumps(row)+'\n');stream.flush()


def report(folder):
    if not (folder/'run_config.json').exists():print(folder.name,'not prepared.');return
    from rescore_audio_steering import paired
    plan=json.loads((folder/'run_config.json').read_text())
    rows=list(answers(plan,folder).values())
    if not rows:print(plan['cohort'],'no answers yet.');return
    scored,provenance=winners.score_rows(dict(target=plan['target']),rows,folder)
    groups={(p,c):{(r['question_id'],r['voice']):r for r in scored if r['prompt_key']==p and r['condition']==c}
            for p in plan['prompts'] for c in CONDITIONS}
    results=[]
    for prompt in plan['prompts']:
        for baseline,test in (('neutral','angry-vector'),('neutral','happy-vector'),('angry-vector','happy-vector')):
            a,b=groups[prompt,baseline],groups[prompt,test]
            ids=sorted(a.keys()&b.keys())
            if not ids:continue
            for metric in a[ids[0]]['metrics']:
                value=paired([(a[q]['metrics'][metric],b[q]['metrics'][metric]) for q in ids])
                results.append(dict(prompt_key=prompt,baseline=baseline,test=test,metric=metric,**value))
                print(f'{plan["cohort"]}/{prompt}: {test} vs {baseline} / {metric}: N={len(ids)}; gain={value["gain_pp"]:+.2f} pp; p={value["mcnemar_exact_p"]:.4g}; CI={value["ci95_gain_pp"]}')
    prompt_effects=[]
    for condition in CONDITIONS:
        a,b=groups['concise',condition],groups['older',condition]
        ids=sorted(a.keys()&b.keys())
        if not ids:continue
        for metric in a[ids[0]]['metrics']:
            prompt_effects.append(dict(condition=condition,metric=metric,**paired([(a[q]['metrics'][metric],b[q]['metrics'][metric]) for q in ids])))
    lengths={p+'/'+c:dict(n=len(g),mean=statistics.mean(r['response_tokens'] for r in g.values() if r.get('response_tokens') is not None))
             for (p,c),g in groups.items() if g and all(r.get('response_tokens') is not None for r in g.values())}
    winners.atomic_text(folder/'summary.json',json.dumps(dict(cohort=plan['cohort'],paired=results,prompt_effects=prompt_effects,
        lengths=lengths,scorer=provenance,note='Gains relative to each prompt baseline. Direct happy-minus-angry and paired older-minus-concise comparisons. Exploratory, unadjusted.'),indent=2)+'\n')
    winners.atomic_text(folder/'rescored.jsonl',''.join(json.dumps(r)+'\n' for r in scored))


def report_common(root):
    """Compare original/shared3 vectors only on questions held out from BOTH fittings."""
    from rescore_audio_steering import paired
    rows={}
    plans={}
    for cohort in COHORTS:
        folder=root/cohort
        if not (folder/'rescored.jsonl').exists():return
        plans[cohort]=json.loads((folder/'run_config.json').read_text())
        rows[cohort]={(r['question_id'],r['voice'],r['prompt_key'],r['condition']):r for r in read_rows(folder/'rescored.jsonl')}
    common=set(plans['original']['test_ids'])&set(plans['shared3']['test_ids'])
    if common & (set(plans['original']['collection_ids'])|set(plans['shared3']['collection_ids'])):
        raise ValueError('Common comparison includes a question used to fit a vector.')
    results=[]
    for prompt in plans['original']['prompts']:
        for condition in CONDITIONS:
            keys=sorted(k for k in rows['original'] if k in rows['shared3'] and k[0] in common and k[2:]==(prompt,condition))
            if not keys:continue
            for k in keys:
                if any(rows['original'][k][f]!=rows['shared3'][k][f] for f in ('audio_sha256','prompt','max_new_tokens')):
                    raise ValueError('Cross-cohort vector comparison needs identical neutral audio, prompt and budget.')
            for metric in rows['original'][keys[0]]['metrics']:
                result=paired([(rows['original'][k]['metrics'][metric],rows['shared3'][k]['metrics'][metric]) for k in keys])
                results.append(dict(prompt_key=prompt,condition=condition,metric=metric,comparison='shared3 minus original',**result))
                print(f'COMMON HELD-OUT / {prompt}/{condition}/{metric}: N={len(keys)}; shared3-original={result["gain_pp"]:+.2f} pp; p={result["mcnemar_exact_p"]:.4g}')
    winners.atomic_text(root/'common-heldout-comparison.json',json.dumps(dict(eligible_common_ids=sorted(common),paired=results,
        note='Only question IDs held out from BOTH fittings. No outcome filtering. Neutral comparisons disclose baseline differences. '
             'Other comparisons are paired raw accuracy differences for the named vector condition; see each cohort report for baseline-relative gains.'),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','run','summary'),default='run')
    parser.add_argument('--cohorts',choices=COHORTS,nargs='+',default=list(COHORTS))
    parser.add_argument('--original-root',type=Path,default=Path('outputs/diffa-happy-winners/gsm8k'))
    parser.add_argument('--shared-root',type=Path,default=Path('outputs/diffa-gsm-shared-three-zeroshot-hashes'))
    parser.add_argument('--output',type=Path,default=Path('outputs/diffa-gsm-prompt-ablation'))
    args=parser.parse_args()
    if args.mode=='summary':
        for cohort in args.cohorts:report(args.output/cohort)
        report_common(args.output)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'queue.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plans={}
        for cohort in args.cohorts:
            root=args.original_root if cohort=='original' else args.shared_root
            try:plans[cohort]=prepare(cohort,root,args.output/cohort)
            except FileNotFoundError as error:
                raise FileNotFoundError('Restore the completed original GSM angry/happy vectors using pack_diffa_gsm_prompt_inputs.py on original DIFFA, '
                    'then restore_diffa_bundle.py the supplemental ZIP here. Existing shared3 vectors are also required.') from error
            done=reuse(plans[cohort],args.output/cohort,[args.output/c for c in plans])
            print(cohort,'answers reused:',len(done),'; pending:',len(pending(plans[cohort],done)),flush=True)
        if args.mode=='prepare':return
        with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
            print('Waiting for the active DIFFA queue; no model loaded while waiting.',flush=True)
            fcntl.flock(gpu,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(30)
            model=tokenizer=features=None
            for cohort,plan in plans.items():
                folder=args.output/cohort
                # Refresh reuse after waiting: the main queue may have finished more concise answers.
                done=reuse(plan,folder,[args.output/c for c in plans])
                if pending(plan,done):
                    for d in plan['vectors'].values():
                        if winners.digest(d['vector'])!=d['vector_sha256']:raise ValueError('Fixed vector changed after preparation.')
                    if model is None:
                        from extract_vectors import load_model
                        model,tokenizer,features=load_model()
                        from diffa_model_receipts import model_revisions
                        winners.atomic_text(args.output/'runtime.json',json.dumps(dict(
                            runner_sha256=winners.digest(__file__),generation_sha256=winners.digest(Path(__file__).with_name('extract_vectors.py')),
                            hook_sha256=winners.digest(Path(__file__).with_name('steer.py')),model_revisions=model_revisions(),
                            model_config=model.config.to_dict()),indent=2,default=str)+'\n')
                    generate(plan,folder,model,tokenizer,features)
                report(folder)
            report_common(args.output)


if __name__=='__main__':main()
