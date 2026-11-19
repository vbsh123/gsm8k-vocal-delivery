"""Prepare SVAMP/BBH3/BBH5; later run own-vector Phi steering at an explicitly chosen layer."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from types import SimpleNamespace

import evaluate_gsm_candidates as gsm
import phi4_all_layer_diagnostic as native_hooks
import run_reasoning_transfer as reasoning
import run_svamp500 as svamp
import run_bbh_logical200 as bbh3
import run_kimi_bbh5 as bbh5
import steer_phi4_gsm_reverse as engine
from run_kimi_gsm8k import prepare_audio
from kimi_bbh_layer_diagnostic import digest,freeze
from pack_experiment_backup import snapshot
from steer import read_rows

TASKS={
    'gsm8k':dict(half=400),
    'multiarith':dict(half=290),
    'svamp':dict(filename='SVAMP-source.json',url=svamp.SOURCE_URL,sha=svamp.SOURCE_SHA256,half=500,
        sources=['azure-svamp1000','azure-svamp500','azure-svamp500-hashes','azure-svamp-remaining500']),
    'bbh3':dict(filename='logical_deduction_three_objects-source.json',url=bbh3.SOURCE_URL,sha=bbh3.SOURCE_SHA256,half=125,
        sources=['azure-bbh-three-models-regenerated','azure-bbh-logical250-steering','azure-bbh-logical200']),
    'bbh5':dict(filename='logical_deduction_five_objects-source.json',url=bbh5.SOURCE_URL,sha=bbh5.SOURCE_SHA256,half=125,
        sources=['azure-kimi-bbh-five250','azure-bbh-five250'])}


def questions(task,data):
    if task=='svamp':return svamp.select_questions(data,1000),gsm.ORIGINAL
    if task=='bbh3':return bbh3.select_questions(data,250),bbh3.PROMPT
    rows=reasoning.bbh_questions(data,'five');random.Random(42).shuffle(rows)
    return rows,bbh5.PROMPT


def prepare_data(task,root):
    spec=TASKS[task];folder=root/task
    if task in ('gsm8k','multiarith'):
        folder.mkdir(parents=True,exist_ok=True)
        if task=='gsm8k':
            path=Path('outputs/phi4-gsm400-all-layers-test869-minus1/run_config.json')
            original=json.loads(path.read_text());collection=original['collection']
            test=[dict(question_id=r['question_id'],voice=r['voice'],neutral=r) for r in original['recordings'] if r['style']=='neutral']
            bank=original['bank'];prompt=original['prompt']
        else:
            from phi4_reverse_followups import multi_selection
            collection,records,bank=multi_selection(Path('outputs/azure-multiarith580'))
            test=[dict(question_id=r['question_id'],voice=r['voice'],neutral=r) for r in records];prompt=gsm.ORIGINAL
        expected=(400,869) if task=='gsm8k' else (290,290)
        train_ids={p['question_id'] for p in collection};test_ids={p['question_id'] for p in test}
        if (len(collection),len(test))!=expected or len(train_ids)!=expected[0] or len(test_ids)!=expected[1] or train_ids & test_ids:
            raise ValueError('Unexpected or overlapping frozen collection/test split.')
        data=dict(protocol='phi4-next-task-data-v1',task=task,prompt=prompt,seed=42,
            collection=collection,test=test,bank=bank,source='existing frozen experiment split')
        freeze(folder/'data_config.json',data)
        print(f'{task}: reused {len(collection)} collection /{len(test)} test; no new TTS.',flush=True)
        return data
    source=folder/'audio';source.mkdir(parents=True,exist_ok=True)
    path=folder/spec['filename'];candidates=[Path('outputs')/p for p in spec['sources']]
    if not path.exists():
        cached=next((p/spec['filename'] for p in candidates if (p/spec['filename']).is_file()),None)
        data=cached.read_bytes() if cached else reasoning.download(spec['url'])
        if hashlib.sha256(data).hexdigest()!=spec['sha']:raise ValueError('Official task source differs.')
        path.write_bytes(data)
    if digest(path)!=spec['sha']:raise ValueError('Cached source checksum differs.')
    rows,prompt=questions(task,path.read_bytes());half=spec['half']
    if len(rows)!=2*half or len({q['question_id'] for q in rows})!=len(rows):raise ValueError('Unexpected task count.')
    freeze(folder/'selection.json',dict(task=task,seed=42,collection_ids=[q['question_id'] for q in rows[:half]],
        test_ids=[q['question_id'] for q in rows[half:]],source_sha256=spec['sha'],prompt=prompt,
        rule='Same seeded half/half selection, no outcome/classifier filtering; no layer/strength chosen during preparation.'))
    reasoning.bank_file(source/'questions.jsonl',rows)
    prepare_audio(SimpleNamespace(source=source,audio_sources=candidates),dict(collection=[],test=rows))
    records=reasoning.source_records(source,{q['question_id']:q for q in rows},['neutral','angry'])
    pairs=[]
    for q in rows:
        pair=dict(question_id=q['question_id'])
        for style in ('neutral','angry'):
            row=dict(records[q['question_id'],style],reference_answer=q['answer']);row.pop('answer')
            pair[style]=row
        if pair['neutral']['voice']!=pair['angry']['voice']:raise ValueError('Voices differ.')
        pair['voice']=pair['neutral']['voice'];pairs.append(pair)
    data=dict(protocol='phi4-next-task-data-v1',task=task,source_sha256=spec['sha'],prompt=prompt,seed=42,
        collection=pairs[:half],test=pairs[half:],bank={q['question_id']:q for q in rows},source=str(source.absolute()))
    freeze(folder/'data_config.json',data)
    print(f'{task}: READY {half} collection /{half} test questions; audio and split only, no model loaded.',flush=True)
    return data


def make_plan(data,data_path,layer,group,strength):
    label=f'neutral-{data["task"]}-L{layer}-{group}-a{strength:g}'
    tasks=[]
    for pair in data['test']:
        for condition,style,amount in [('neutral','neutral',0.),('angry','angry',0.),(label,'neutral',strength)]:
            if style in pair:tasks.append(dict(pair[style],condition=condition,strength=amount))
    return dict(protocol='phi4-task-own-mean-steering-v1',task=data['task'],data_config_sha256=digest(data_path),
        recipe=dict(gsm.MODELS['phi4'],dtype='bfloat16',do_sample=False,num_beams=1,use_cache=True,repetition_penalty=1.,text_only_output=True),
        collection=data['collection'],bank={p['question_id']:data['bank'][p['question_id']] for p in data['test']},
        recordings=tasks,prompt=data['prompt'],seed=42,max_new_tokens=512,layer=layer,group=group,strength=strength,
        vector_definition='Own-task angry-minus-neutral raw mean on collection half. Answer: all native generation '
            'last-prediction forwards including prefill. Audio: native audio positions on prefill only. Equal per-question mean.',
        selection='Frozen half/half data split; whole test half evaluated, no recovery/outcome/classifier filter. '
            'Layer/group/strength supplied explicitly after GSM diagnostics; original task delivery results may have been inspected.')


class Native:
    def __init__(self,plan):self.native=engine.NativePhi(plan['layer']);self.plan=plan
    def generate(self,row,prompt,direction=None,strength=0.,observe=False):
        trace=native_hooks.generate(self.native,row,prompt,collect=observe,target=None if observe or not strength else self.plan['layer'],
            group=self.plan['group'],direction=direction,strength=strength,capture_layers=[self.plan['layer']] if observe else None)
        if observe:
            trace['mean']=trace.pop('means')[0,native_hooks.GROUPS.index(self.plan['group'])]
        return trace


def completed(plan,output,repair=False):
    tasks={(r['question_id'],r['voice'],r['condition']):r for r in plan['recordings']};done={};config_hash=digest(output/'run_config.json')
    for r in read_rows(output/'results.jsonl',repair=repair):
        key=r['question_id'],r['voice'],r['condition']
        if (key not in tasks or key in done or r['run_config_sha256']!=config_hash or r['prompt']!=plan['prompt']
                or r['audio_sha256']!=tasks[key]['audio_sha256'] or r['seed']!=42 or r['max_new_tokens']!=512
                or r['strength']!=tasks[key]['strength'] or r['reference_answer']!=tasks[key]['reference_answer']
                or r['model']!=plan['recipe']['model'] or r['revision']!=plan['recipe']['revision']):raise ValueError('Saved steering answer differs.')
        done[key]=r
    return tasks,done


def reuse(plan,output):
    if plan['task']=='svamp':return
    source=Path('outputs')/('bbh-three-candidates' if plan['task']=='bbh3' else 'bbh-five-candidates')/'phi4/original'
    config=source/'run_config.json';results=source/'results.jsonl'
    if not config.exists() or not results.exists():return
    old=json.loads(config.read_text())
    if old['recipe']!=plan['recipe'] or old['prompt']!=plan['prompt']:return
    tasks,done=completed(plan,output,repair=True);data,_=snapshot(results)
    with (output/'results.jsonl').open('a') as stream:
        for line in data.splitlines():
            if not line.strip():continue
            row=json.loads(line);key=row['question_id'],row['voice'],row.get('condition',row['style'])
            if key not in tasks or key in done or key[2] not in ('neutral','angry'):continue
            task=tasks[key]
            if (row['audio_sha256']!=task['audio_sha256'] or row['prompt']!=plan['prompt'] or row['seed']!=42
                    or row['max_new_tokens']!=512 or row['interventions'] or row['reference_answer']!=task['reference_answer']
                    or row['model']!=plan['recipe']['model'] or row['revision']!=plan['recipe']['revision']):continue
            saved=dict(task,response=row['response'],response_tokens=row['response_tokens'],prompt=plan['prompt'],
                seed=42,max_new_tokens=512,run_config_sha256=digest(output/'run_config.json'),reused_from=str(source),
                model=plan['recipe']['model'],revision=plan['recipe']['revision'],interventions=[])
            stream.write(json.dumps(saved)+'\n');done[key]=saved


def collection_vector(plan,output,native,root):
    """Collect once per task/layer/group, reusing the same direction across strengths."""
    import torch
    from kimi_bbh_layer_diagnostic import atomic_save
    cache=root/plan['task']/f"L{plan['layer']}-{plan['group']}";cache.mkdir(parents=True,exist_ok=True)
    common={k:plan[k] for k in ('task','collection','recipe','prompt','seed','max_new_tokens','layer','group','vector_definition')}
    common['protocol']='phi4-sweep-collection-v1';freeze(cache/'run_config.json',common)
    if plan['task']=='gsm8k':
        source=Path('outputs/phi4-gsm400-all-layers-test869-minus1')
        original=json.loads((source/'run_config.json').read_text())
        for key in ('collection','recipe','prompt'):
            if original[key]!=plan[key]:raise ValueError('GSM source collection recipe differs: '+key)
        saved=torch.load(source/'style-vectors.pt',map_location='cpu',weights_only=True)
        if saved['run_config_sha256']!=digest(source/'run_config.json') or saved['collection_question_ids']!=[p['question_id'] for p in plan['collection']]:
            raise ValueError('GSM source vector provenance differs.')
        if saved['groups']!=['answer','audio'] or saved['angry_minus_neutral'].shape!=native_hooks.SHAPE or saved['recipe']!=plan['recipe'] or not torch.isfinite(saved['angry_minus_neutral']).all():raise ValueError('GSM source vector layout differs.')
        vector=saved['angry_minus_neutral'][plan['layer'],saved['groups'].index(plan['group'])]
        provenance=dict(source=str(source/'style-vectors.pt'),sha256=digest(source/'style-vectors.pt'))
    else:
        vector=engine.collect(common,cache,native,digest(cache/'run_config.json'))
        provenance=dict(source=str(cache/'style-vectors.pt'),sha256=digest(cache/'style-vectors.pt'))
    state=dict(protocol=plan['protocol'],run_config_sha256=digest(output/'run_config.json'),layer=plan['layer'],group=plan['group'],
        collection_question_ids=[p['question_id'] for p in plan['collection']],model=plan['recipe']['model'],revision=plan['recipe']['revision'],
        angry_minus_neutral=vector,source_vector=provenance)
    path=output/'style-vectors.pt'
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=True)
        if old['run_config_sha256']!=state['run_config_sha256'] or not torch.equal(old['angry_minus_neutral'],vector):raise ValueError('Sweep vector changed.')
    else:atomic_save(path,state)
    return vector


def evaluate(plan,output,collection_root=None):
    import torch
    reuse(plan,output);tasks,done=completed(plan,output,repair=True)
    if len(done)==len(tasks):print('All answers complete; no model loaded.');return
    from evaluate_bbh_candidates import concurrent_budget_now
    concurrent_budget_now(18.)
    native=Native(plan)
    (output/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,transformers=engine.version('transformers'),
        gpu=torch.cuda.get_device_name(0),script_sha256=digest(__file__),native_hooks_sha256=digest(native_hooks.__file__),
        policy='No GPU queue/lock;18 GiB Torch cap'),indent=2)+'\n')
    vector=collection_vector(plan,output,native,collection_root) if collection_root else engine.collect(plan,output,native,digest(output/'run_config.json'))
    for i,(key,row) in enumerate(tasks.items(),1):
        if key in done:continue
        print(f'{plan["task"]} [{i}/{len(tasks)}] {row["question_id"]} {row["condition"]}',flush=True)
        trace=native.generate(row,plan['prompt'],vector if row['strength'] else None,row['strength'])
        saved=dict(row,**trace,prompt=plan['prompt'],seed=42,max_new_tokens=512,
            run_config_sha256=digest(output/'run_config.json'),model=plan['recipe']['model'],revision=plan['recipe']['revision'],
            vector_sha256=digest(output/'style-vectors.pt') if row['strength'] else None,
            interventions=[dict(layer=plan['layer'],group=plan['group'],strength=row['strength'])] if row['strength'] else [])
        with (output/'results.jsonl').open('a') as stream:stream.write(json.dumps(saved)+'\n')
        print(trace['response'],flush=True)


def report(output):
    from rescore_audio_steering import HarnessTaskScorer,numeric_equal,numeric_within_tolerance,paired
    from evaluate_bbh_audio_models import strict_format_bits
    path=output/'run_config.json'
    if not path.exists():print(output,'not run/prepared.');return
    plan=json.loads(path.read_text());_,done=completed(plan,output);rows=list(done.values())
    print(f'{plan["task"]}: {len(rows)}/{len(plan["recordings"])} answers; L{plan["layer"]} {plan["group"]} x{plan["strength"]}.')
    if not rows:return
    math_task=plan['task'] in ('gsm8k','svamp','multiarith')
    print('Math: installed GSM8K filters/metric on dataset references; numeric equivalence additional. SVAMP/MultiArith are spoken adaptations using these filters.' if math_task else 'BBH: installed upstream task scoring on saved native audio-adapted responses.')
    name='gsm8k' if math_task else 'bbh_cot_zeroshot_logical_deduction_'+('three' if plan['task']=='bbh3' else 'five')+'_objects'
    scorer=HarnessTaskScorer(name,output/'harness/upstream')
    docs=[dict(question=plan['bank'][r['question_id']]['question'],answer=r['reference_answer']) if math_task else
          dict(input=plan['bank'][r['question_id']]['original_input'],target=plan['bank'][r['question_id']]['original_target']) for r in rows]
    values=scorer.score([r['response'] for r in rows],docs)
    bits=None if math_task else strict_format_bits(scorer,[r['response'] for r in rows],docs,'ABC' if plan['task']=='bbh3' else 'ABCDE')
    for i,r in enumerate(rows):
        r['scores']={m:v[i]['correct'] for m,v in values.items()}
        r['scores']['format-compliant']=values['strict-match'][i]['prediction']!='[invalid]' if math_task else bits[i]
        if math_task:
            pred=values['flexible-extract'][i]['prediction']
            r['scores']['flexible-numeric']=numeric_equal(pred,r['reference_answer'])
            if plan['task']=='svamp':r['scores']['author-tolerance']=numeric_within_tolerance(pred,r['reference_answer'])
    labels=list(dict.fromkeys(r['condition'] for r in plan['recordings']));groups={c:{(r['question_id'],r['voice']):r for r in rows if r['condition']==c} for c in labels}
    keys=sorted(set.intersection(*(set(g) for g in groups.values())));accuracy=[];comparisons=[]
    print(len(keys),f'questions completed in all {len(groups)} conditions; unadjusted paired tests.')
    if keys:
        for c,g in groups.items():
            lengths=[g[k]['response_tokens'] for k in keys]
            print(c,'mean/median tokens:',round(statistics.mean(lengths),1),statistics.median(lengths))
    for metric in rows[0]['scores']:
        for c,group in groups.items():
            if not keys:continue
            correct=sum(group[k]['scores'][metric] for k in keys)
            print(f'{metric} / {c}: {correct}/{len(keys)} ({correct/len(keys):.1%})')
            accuracy.append(dict(metric=metric,condition=c,n=len(keys),correct=correct))
            if c!='neutral':
                result=paired([(groups['neutral'][k]['scores'][metric],group[k]['scores'][metric]) for k in keys])
                comparisons.append(dict(metric=metric,condition=c,**result));print(f'  gain={result["gain_pp"]:+.2f}; p={result["mcnemar_exact_p"]:.4g}; CI={result["ci95_gain_pp"]}')
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (output/'summary.json').write_text(json.dumps(dict(accuracy=accuracy,paired=comparisons,scorer=scorer.provenance,
        note='Native spoken adaptation, upstream saved-answer scoring; collection/test disjoint; tests unadjusted.'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','run','summary'),default='prepare')
    p.add_argument('--task',choices=('all',*TASKS),default='all')
    p.add_argument('--data-root',type=Path,default=Path('outputs/phi4-next-tasks'))
    p.add_argument('--output-root',type=Path,default=Path('outputs/phi4-task-steering'))
    p.add_argument('--layer',type=int)
    p.add_argument('--group',choices=('answer','audio'),default='answer')
    p.add_argument('--strength',type=float,default=-1.)
    args=p.parse_args()
    tasks=list(TASKS) if args.task=='all' else [args.task]
    if args.mode=='prepare':
        for task in tasks:prepare_data(task,args.data_root)
        print('Prepared only. Choose layer/group/strength explicitly for later --mode run.');return
    if args.mode=='run' and args.task=='all':p.error('Use --task for one model process, or run_phi4_prepared_tasks.sh for all three.')
    if args.layer is None or not 0<=args.layer<32 or not math.isfinite(args.strength):p.error('Supply valid --layer0..31 and finite strength.')
    for task in tasks:
        output=args.output_root/task/f'L{args.layer}-{args.group}-a{args.strength:g}'
        if args.mode=='summary':report(output);continue
        data_path=args.data_root/task/'data_config.json';data=json.loads(data_path.read_text())
        plan=make_plan(data,data_path,args.layer,args.group,args.strength);output.mkdir(parents=True,exist_ok=True)
        freeze(output/'run_config.json',plan);evaluate(plan,output)


if __name__=='__main__':main()
