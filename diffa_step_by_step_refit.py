"""GSM/MultiArith original+shared3 cohorts: older step-by-step sentence, deliveries, fresh vectors and +1 steering."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

import diffa_happy_nightly as task
from jenny_bundle_registry import Registry
from steer import read_rows

PROTOCOL='diffa-original-shared3-step-by-step-refit-v1'
SENTENCE='Give concise step-by-step reasoning with only the necessary calculations.'
DATASETS=('gsm8k','multiarith')
COHORTS=('original','shared3')
EMOTIONS=('angry','happy')


def step_prompt(prompt):
    if prompt.count(SENTENCE)!=1:
        raise ValueError('Expected exactly one explicit calculation/reasoning sentence; no other prompt edits allowed.')
    return prompt.replace(SENTENCE, "Let's think step by step.", 1)


def prepare(name,cohort,sources,registry,output):
    templates={e:json.loads(sources[e].read_text()) for e in EMOTIONS}
    base=templates['happy'];train,test=base['collection_ids'],base['test_ids']
    if (base['name']!=name or base['seed']!=42 or not train or not test
            or len(set(train+test))!=len(train+test)
            or [r['question_id'] for r in base['target']['records']]!=test):
        raise ValueError('Invalid frozen cohort; original question IDs must not be resplit.')
    for e,t in templates.items():
        if (t['collection_ids']!=train or t['test_ids']!=test or t['prompt']!=base['prompt']
                or t['collection_prompt']!=base['collection_prompt'] or t['max_new_tokens']!=base['max_new_tokens']
                or [p['question_id'] for p in t['pairs']]!=train
                or [(r['question_id'],r['audio_sha256']) for r in t['target']['records']] !=
                   [(r['question_id'],r['audio_sha256']) for r in base['target']['records']]
                or [p['neutral']['audio_sha256'] for p in t['pairs']]!=[p['neutral']['audio_sha256'] for p in base['pairs']]):
            raise ValueError('Angry/happy source cohorts or prompts differ.')
    bank=deepcopy(base['bank']);bank.update(base['target']['bank'])
    expected={(r['question_id'],'neutral'):r['audio_sha256'] for r in base['target']['records']}
    for e,t in templates.items():
        for pair in t['pairs']:
            expected[pair['question_id'],'neutral']=pair['neutral']['audio_sha256']
            expected[pair['question_id'],e if cohort=='shared3' else 'happy']=pair['emotion']['audio_sha256']
        for r in t.get('recordings',[]):
            style=r.get('condition',r.get('style'))
            if style in ('neutral','angry','happy') and r['question_id'] in test:
                expected.setdefault((r['question_id'],style),r['audio_sha256'])
    records={}
    for q in train+test:
        example=bank[q];answer=task.normal_answer(example['answer'])
        for style in ('neutral','angry','happy'):
            item=registry.choose(name,q,style,question=example['question'],answer=answer,expected=expected.get((q,style)))
            if item is None:
                raise FileNotFoundError(f'Missing exact {name}/{q}/{style} audio. No TTS, classifier filtering or question substitution.')
            options=registry.options.get((name,q,style),[])
            variants={x['record']['audio_sha256'] for x in options if x['question']==example['question'] and x['answer']==answer}
            if (q,style) not in expected and len(variants)>1:
                raise ValueError(f'Ambiguous recording for {name}/{q}/{style}; need an exact source hash.')
            records[q,style]=item['record']
    plans={}
    for e in EMOTIONS:
        original=templates[e]
        target=deepcopy(original['target'])
        target.update(prompt=step_prompt(original['prompt']),records=[records[q,'neutral'] for q in test],
                      bank={q:bank[q] for q in test},extraction_ids=train)
        plan=dict(deepcopy(original),protocol=PROTOCOL,emotion=e,name=name,cohort=cohort,bank=bank,target=target,
                  collection_ids=train,test_ids=test,prompt=target['prompt'],collection_prompt=step_prompt(original['collection_prompt']),
                  pairs=[dict(question_id=q,neutral=records[q,'neutral'],emotion=records[q,e]) for q in train],
                  recordings=[dict(records[q,s],condition=s) for q in test for s in ('neutral',e)],
                  parent_config=str(sources[e].absolute()),parent_config_sha256=task.winners.digest(sources[e]),
                  averaging='Matched forwards within each pair; token-position means; equal per-question weight.',
                  selection='Exact frozen original/shared3 fitting and evaluation IDs. Shared3 membership inherited; no new filtering or resplitting.',
                  note='Replace only the reasoning sentence with the older Let us think step by step instruction in collection and evaluation. Keep format, budget, seed and audio. '
                       'Fresh emotion-minus-neutral collection, natural delivery evaluation, then L31 answer +1 steering.')
        folder=output/e;folder.mkdir(parents=True,exist_ok=True)
        task.winners.freeze(folder/'run_config.json',plan)
        delivery=folder/'delivery';delivery.mkdir(exist_ok=True)
        task.winners.freeze(delivery/'run_config.json',dict(protocol=PROTOCOL,target=target,emotion=e,conditions=['neutral',e]))
        plans[e]=plan
    print(f'{name}/{cohort}: EXACT {len(train)}/{len(test)} collection/evaluation IDs. Fresh angry/happy vectors; older step-by-step instruction retained.',flush=True)
    return plans


def collect_both(plans,output,model,tokenizer,features):
    """One neutral capture per fitting question, paired separately with angry and happy."""
    import torch
    import diffa_stream_collection as running
    import src.fast_dllm.diffa_generate as decoder
    ids=plans['angry']['collection_ids']
    if ids!=plans['happy']['collection_ids'] or plans['angry']['collection_prompt']!=plans['happy']['collection_prompt']:
        raise ValueError('Paired collection must share exact IDs and prompt.')
    states={e:running.load(plans[e],output/e,task.winners.digest(output/e/'run_config.json')) for e in EMOTIONS}
    for index,q in enumerate(ids):
        pending=[e for e in EMOTIONS if len(states[e]['completed_ids'])<=index]
        if not pending:continue
        if any(len(states[e]['completed_ids'])!=index for e in pending):raise ValueError('Non-prefix collection checkpoint.')
        base=plans['angry'];record=base['pairs'][index]['neutral']
        print(f'{base["name"]}/{base["cohort"]} collect [{index+1}/{len(ids)}] {q}; neutral once + {pending}',flush=True)
        nr,ne,n=task.emotion.capture(model,lambda:task.generate(record,base,base['collection_prompt'],model,tokenizer,features),decoder,base['max_new_tokens'])
        for e in pending:
            plan=plans[e];record=plan['pairs'][index]['emotion']
            response,encoding,h=task.emotion.capture(model,lambda:task.generate(record,plan,plan['collection_prompt'],model,tokenizer,features),decoder,plan['max_new_tokens'])
            nm,hm,delta,matched=task.average_pair(n,h,'matched')
            saved=dict(protocol=PROTOCOL,run_config_sha256=task.winners.digest(output/e/'run_config.json'),question_id=q,
                       neutral=nm,emotion=hm,difference=delta,neutral_response=nr,emotion_response=response,
                       neutral_forwards=n.shape[1],emotion_forwards=h.shape[1],matched_forwards=matched,
                       neutral_encoding=ne,emotion_encoding=encoding)
            running.append(states[e],saved)
            task.save_tensor(torch,states[e],output/e/'collection-stream.pt')
            del h,nm,hm,delta,saved
        del n
    for e in EMOTIONS:running.finalize(plans[e],output/e,states[e])


def deliveries(plan,folder,repair=False):
    expected={(r['question_id'],r['voice'],r['condition']):r for r in plan['recordings']}
    result={}
    for row in read_rows(folder/'results.jsonl',repair=repair):
        k=row['question_id'],row['voice'],row['condition']
        if (k not in expected or k in result or row.get('interventions') or float(row.get('strength') or 0.)!=0.
                or not task.winners.reusable(row,{},expected[k],plan['target'])):
            raise ValueError('Cached older-step-by-step delivery differs from its exact audio/prompt/recipe.')
        result[k]=row
    return result


def reuse_deliveries(plan,folder,other_folders):
    done=deliveries(plan,folder,repair=True)
    expected={(r['question_id'],r['voice'],r['condition']):r for r in plan['recordings']}
    with (folder/'results.jsonl').open('a') as stream:
        for source in other_folders:
            if source==folder or not (source/'results.jsonl').exists():continue
            for row in read_rows(source/'results.jsonl'):
                k=row['question_id'],row['voice'],row['condition']
                if k not in expected or k in done or row.get('interventions') or float(row.get('strength') or 0.)!=0.:continue
                if not task.winners.reusable(row,{},expected[k],plan['target']):continue
                copied=dict(expected[k],**{f:row[f] for f in ('response','strength','prompt','seed','max_new_tokens','steps','block_length','temperature','decoder','interventions')},
                            response_tokens=row.get('response_tokens'),audio_encoding=row.get('audio_encoding'),reused_from=str(source))
                stream.write(json.dumps(copied)+'\n');done[k]=copied


def steering_job(plan,folder):
    source=task.descriptor(plan,folder)
    return dict(protocol=PROTOCOL,kind='fresh-step-by-step-steering',source=source,target=source,strengths=[0.,1.],
                layer=31,group='answer',schedule='always',mode='fixed')


def compare_emotions(folder):
    from rescore_audio_steering import paired
    for stage in ('delivery','strength-1'):
        groups={}
        for e in EMOTIONS:
            path=folder/e/stage/'rescored.jsonl'
            if not path.exists():break
            rows=read_rows(path)
            groups[e]={r['question_id']:r for r in rows if (r['condition']==e if stage=='delivery' else r.get('strength')==1.)}
        if len(groups)!=2:continue
        ids=sorted(groups['angry'].keys()&groups['happy'].keys())
        if not ids:continue
        values=[]
        for metric in groups['angry'][ids[0]]['metrics']:
            result=paired([(groups['angry'][q]['metrics'][metric],groups['happy'][q]['metrics'][metric]) for q in ids])
            values.append(dict(metric=metric,**result))
            print(f'{folder}: {stage} HAPPY vs ANGRY / {metric}: N={len(ids)}; gain={result["gain_pp"]:+.2f} pp; p={result["mcnemar_exact_p"]:.4g}')
        task.winners.atomic_text(folder/f'{stage}-happy-vs-angry.json',json.dumps(dict(question_ids=ids,paired=values),indent=2)+'\n')


def summarize(output,names,cohorts):
    for name in names:
        for cohort in cohorts:
            folder=output/cohort/name
            for e in EMOTIONS:
                path=folder/e/'run_config.json'
                if not path.exists():print(name,cohort,e,'not prepared.');continue
                plan=json.loads(path.read_text())
                task.report_collection(plan,folder/e)
                task.report_delivery(plan,folder/e/'delivery')
                if (folder/e/'strength-1/run_config.json').exists():task.report_steering(folder/e/'strength-1')
            compare_emotions(folder)
        compare_protocols(output,name)


def compare_protocols(output,name):
    from rescore_audio_steering import paired
    roots={c:output/c/name for c in COHORTS}
    paths={c:root/'angry/run_config.json' for c,root in roots.items()}
    if not all(p.exists() for p in paths.values()):return
    plans={c:json.loads(p.read_text()) for c,p in paths.items()}
    common=set(plans['original']['test_ids'])&set(plans['shared3']['test_ids'])
    if common&(set(plans['original']['collection_ids'])|set(plans['shared3']['collection_ids'])):
        raise ValueError('Cross-protocol comparison includes fitting questions.')
    results=[]
    for stage in ('delivery','strength-1'):
        for e in EMOTIONS:
            rows={}
            for c in COHORTS:
                path=roots[c]/e/stage/'rescored.jsonl'
                if not path.exists():break
                rows[c]={(r['question_id'],r['voice']):r for r in read_rows(path)
                         if r['question_id'] in common and (r['condition']==e if stage=='delivery' else r.get('strength')==1.)}
            if len(rows)!=2:continue
            ids=sorted(rows['original'].keys()&rows['shared3'].keys())
            if not ids:continue
            for q in ids:
                if any(rows['original'][q][k]!=rows['shared3'][q][k] for k in ('audio_sha256','prompt','max_new_tokens')):
                    raise ValueError('Common-cohort comparison has different audio/prompt/budget.')
            for metric in rows['original'][ids[0]]['metrics']:
                result=paired([(rows['original'][q]['metrics'][metric],rows['shared3'][q]['metrics'][metric]) for q in ids])
                results.append(dict(stage=stage,emotion=e,metric=metric,**result))
                print(f'{name} COMMON HELD-OUT / {stage}/{e}/{metric}: N={len(ids)}; shared3-original={result["gain_pp"]:+.2f} pp; p={result["mcnemar_exact_p"]:.4g}')
    task.winners.atomic_text(output/f'common-heldout-{name}.json',json.dumps(dict(eligible_common_ids=sorted(common),paired=results,
        note='Paired shared3-minus-original accuracy on IDs held out from BOTH fittings; no outcome filtering. '
             'Emotion comparison within each protocol is reported separately.'),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','run','summary'),default='run')
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    parser.add_argument('--cohorts',nargs='+',choices=COHORTS,default=list(COHORTS))
    parser.add_argument('--original-root',type=Path,default=Path('outputs/diffa-happy-winners'))
    parser.add_argument('--shared-gsm',type=Path,default=Path('outputs/diffa-gsm-shared-three-zeroshot-hashes'))
    parser.add_argument('--shared-root',type=Path,default=Path('outputs/diffa-shared-three-tasks'))
    parser.add_argument('--audio-root',type=Path,default=Path('.'))
    parser.add_argument('--output',type=Path,default=Path('outputs/diffa-step-by-step-refit'))
    args=parser.parse_args()
    if args.mode=='summary':summarize(args.output,args.datasets,args.cohorts);return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'queue.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('Preparing existing audio lookup on CPU; no TTS, classification or model loading.',flush=True)
        registry=Registry(args.audio_root);cases=[]
        for name in args.datasets:
            for cohort in args.cohorts:
                if cohort=='original':sources={e:args.original_root/name/'run_config.json' for e in EMOTIONS}
                else:
                    root=args.shared_gsm if name=='gsm8k' else args.shared_root/name
                    sources={e:root/e/'run_config.json' for e in EMOTIONS}
                folder=args.output/cohort/name
                plans=prepare(name,cohort,sources,registry,folder)
                for e,plan in plans.items():task.winners.score_rows(dict(target=plan['target']),[],folder/e/'delivery')
                cases.append((folder,plans))
        if args.mode=='prepare':return
        if shutil.disk_usage(args.output).free<700*2**20:raise RuntimeError('Need 700 MiB free for aggregate checkpoints and answers.')
        with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
            print('Waiting for shared DIFFA GPU lock; no model loaded while waiting.',flush=True)
            fcntl.flock(gpu,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(30)
            from extract_vectors import load_model
            from diffa_bbh_angry125 import fill_neutral
            from diffa_model_receipts import model_revisions
            model,tokenizer,features=load_model()
            task.winners.atomic_text(args.output/'runtime.json',json.dumps(dict(runner_sha256=task.winners.digest(__file__),
                collector_sha256=task.winners.digest(task.__file__),model_revisions=model_revisions(),
                generation_sha256=task.winners.digest(Path(__file__).with_name('extract_vectors.py')),
                model_config=model.config.to_dict()),indent=2,default=str)+'\n')
            all_deliveries=[folder/e/'delivery' for folder,_ in cases for e in EMOTIONS]
            # Delivery effects first: see neutral/angry/happy before lengthy refitting.
            for folder,plans in cases:
                for e,plan in plans.items():
                    delivery=folder/e/'delivery'
                    reuse_deliveries(plan,delivery,all_deliveries)
                    fill_neutral(plan,dict(target=plan['target']),delivery,model,tokenizer,features)
                    task.deliver(plan,delivery,model,tokenizer,features)
                    deliveries(plan,delivery)
                    task.report_delivery(plan,delivery,tokenizer)
                compare_emotions(folder)
                compare_protocols(args.output,plans['angry']['name'])
            for folder,plans in cases:
                collect_both(plans,folder,model,tokenizer,features)
                for e,plan in plans.items():
                    task.report_collection(plan,folder/e,tokenizer)
                    out=folder/e/'strength-1';out.mkdir(exist_ok=True)
                    job=steering_job(plan,folder/e)
                    task.winners.freeze(out/'run_config.json',job)
                    task.reuse_neutral(folder/e/'delivery',out)
                    task.winners.generate_job(job,out,model,tokenizer,features)
                    task.report_steering(out,tokenizer)
                compare_emotions(folder)
                compare_protocols(args.output,plans['angry']['name'])


if __name__=='__main__':main()
