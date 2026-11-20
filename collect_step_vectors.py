"""Step-Audio-2-mini: disjoint GSM8K neutral/angry vectors and activation drift.

Observe native full generations; preserve the current evaluation and decoder.
"""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import json
from pathlib import Path
import random
from statistics import mean,median
import subprocess
import time

from bbh_candidate_adapters import MODELS,checkpoint
from extract_audio_vectors import save_tensor
from kimi_bbh_layer_diagnostic import digest,freeze
from run_reasoning_transfer import recording
from steer import read_rows
from step_calculations20 import PROMPT

PROTOCOL='step-gsm-disjoint-all-layer-collection-v1'
GROUPS=['answer','audio']
SOURCES=[Path('outputs')/s for s in ('azure-jenny500-neutral','azure-jenny500-extra','azure-jenny-next500','azure-jenny-remaining-test')]


def select_ids(excluded,evaluation_ids,count):
    ids=sorted(set(excluded));random.Random(42).shuffle(ids)
    if len(ids)!=500 or not 0<count<=len(ids):raise ValueError('Need the original500 excluded collection IDs and a valid pair count.')
    chosen=ids[:count]
    if set(chosen)&set(evaluation_ids):raise ValueError('Collection overlaps Step evaluation.')
    return chosen


def prepare(args):
    import torch
    config=args.output/'run_config.json'
    if config.exists():
        plan=json.loads(config.read_text())
        if plan['protocol']!=PROTOCOL or len(plan['collection_ids'])!=args.pairs or plan['evaluation_run']!=str(args.evaluation.absolute()):raise ValueError('Frozen collection settings changed.')
        return plan
    evaluation=json.loads((args.evaluation/'run_config.json').read_text())
    if evaluation['recipe']['model']!=MODELS['step']['model'] or evaluation['recipe']['revision']!=MODELS['step']['revision'] or evaluation['prompt']!=PROMPT:
        raise ValueError('Need the pinned Step calculation-prompt evaluation config; start/prepare that run first.')
    original=json.loads((args.diffa_source/'run_config.json').read_text())
    vector=Path(original['vector'])
    if digest(vector)!=original['vector_sha256']:raise ValueError('Original DIFFA vector changed; cannot establish collection exclusions.')
    state=torch.load(vector,map_location='cpu',weights_only=True)
    if set(state['extraction_question_ids']) & set(original['bank']) or not set(evaluation['bank'])<=set(original['bank']):raise ValueError('Original collection/test exclusions or Step evaluation cohort differ.')
    ids=select_ids(state['extraction_question_ids'],evaluation['bank'],args.pairs)
    official=read_rows(args.diffa_source/'gsm8k-test.jsonl')
    bank={q:dict(question_id=q,question=official[int(q.rsplit('_',1)[1])]['question'],answer=official[int(q.rsplit('_',1)[1])]['answer'].rsplit('####',1)[-1].strip()) for q in ids}
    audio={}
    for source in args.audio_sources:
        for row in read_rows(source/'plan.jsonl'):
            q=row['question_id'];style=row['style']
            if q not in bank or style not in ('neutral','angry') or row.get('provider')!='azure' or row['voice']!='en-US-JennyNeural':continue
            if row['question']!=bank[q]['question'] or str(row['answer']).rsplit('####',1)[-1].strip()!=bank[q]['answer']:raise ValueError('Audio question/reference mismatch.')
            if not (source/row['audio_path']).is_file():continue
            audio.setdefault((q,style),recording(row,source))
    missing=[(q,s) for q in ids for s in ('neutral','angry') if (q,s) not in audio]
    if missing:raise FileNotFoundError(f'Missing existing collection audio: {missing[:5]}. No TTS requested; pass --audio-sources or restore the clips.')
    pairs=[dict(question_id=q,voice=audio[q,'neutral']['voice'],**{s:audio[q,s] for s in ('neutral','angry')}) for q in ids]
    records=[dict(p[s],condition=s,reference_answer=p[s]['answer']) for p in pairs for s in ('neutral','angry')]
    plan=dict(protocol=PROTOCOL,task='gsm8k',model_key='step',recipe=evaluation['recipe'],prompt=PROMPT,
        prompt_variant='calculations-vector-collection',seed=42,max_new_tokens=512,bank=bank,recordings=records,pairs=pairs,
        collection_ids=ids,evaluation_ids=list(evaluation['bank']),evaluation_run=str(args.evaluation.absolute()),
        evaluation_config_sha256=digest(args.evaluation/'run_config.json'),diffa_source_sha256=digest(args.diffa_source/'run_config.json'),
        official_questions_sha256=digest(args.diffa_source/'gsm8k-test.jsonl'),original_vector_sha256=digest(vector),groups=GROUPS,
        selection=f'Seed42 shuffle of original500 excluded IDs; first{args.pairs}; disjoint from all original718 DIFFA evaluation IDs. No correctness/classifier/duration filtering.',
        capture='All28 Step language-decoder blocks. Audio patch-token means on initial prefill; answer last-position state on every native generation forward, including the initial prompt boundary predicting the first answer token.',
        averaging='Independent mean of last prediction states over each condition full decoding run; angry minus neutral per question, then equal question weight. No matched-forward truncation.',
        scoring='Collection-only descriptive accuracy, not held-out validation. Existing GSM8K filters plus numeric/format diagnostics; unadjusted paired tests.',
        note='Initial-state and full-generation means are separate artifacts. Output lengths/histories differ and can affect generation means. Audio groups are decoder input-audio patch positions, not acoustic encoder layers.')
    freeze(config,plan)
    print(f'Step collection: {len(pairs)} disjoint pairs / {2*len(pairs)} full generations; same calculations prompt, no TTS.',flush=True)
    return plan


@contextmanager
def observe(model):
    """Passive hooks; no return-value replacement or decoder monkeypatching."""
    import torch
    layers=model.model.layers
    state=dict(initial={},totals={},counts={});handles=[];audio_indices=None
    def before(module,args,kwargs):
        nonlocal audio_indices
        if audio_indices is None:
            ids=kwargs.get('input_ids',args[0] if args else None)
            if ids is None or ids.shape[0]!=1:raise ValueError('Expected one native Step input sequence.')
            audio_indices=(ids[0]==151690).nonzero(as_tuple=True)[0]
            if not len(audio_indices):raise ValueError('No Step audio patch tokens found.')
    def hook(index):
        def read(module,args,result):
            h=result[0] if isinstance(result,(tuple,list)) else result
            latest=h[0,-1].detach().float().cpu()
            if index not in state['initial']:
                if audio_indices is None or int(audio_indices.max())>=h.shape[1]:raise ValueError('Native audio positions missing from prefill.')
                audio=h[0,audio_indices].detach().float().mean(0).cpu()
                state['initial'][index]=torch.stack([latest,audio])
                state['totals'][index]=latest.clone();state['counts'][index]=1
            else:
                state['totals'][index]+=latest;state['counts'][index]+=1
        return read
    try:
        handles.append(model.register_forward_pre_hook(before,with_kwargs=True))
        handles.extend(layer.register_forward_hook(hook(i)) for i,layer in enumerate(layers))
        yield state
    finally:
        for handle in handles:handle.remove()


def capture(generator,record):
    import torch
    import numpy as np
    if digest(record['audio_file'])!=record['audio_sha256']:raise ValueError('Frozen collection audio changed.')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    with observe(generator.model) as state:response,tokens=generator(record,PROMPT)
    if len(state['initial'])!=len(generator.model.model.layers) or len(set(state['counts'].values()))!=1:raise ValueError('Missing/misaligned layer observations.')
    initial=torch.stack([state['initial'][i] for i in range(len(state['initial']))])
    averages=torch.stack([state['totals'][i]/state['counts'][i] for i in range(len(state['initial']))])
    pooled=initial.clone();pooled[:,0]=averages
    return dict(initial=initial,mean=pooled,forwards=state['counts'][0],response=response,response_tokens=tokens,
        text_tokens=len(generator.tokenizer.encode(response,add_special_tokens=False)),**generator.last_trace)


def answer_rows(plan,pair,traces):
    for style,trace in traces.items():
        yield dict(pair[style],condition=style,reference_answer=pair[style]['answer'],response=trace['response'],
            response_tokens=trace['response_tokens'],text_tokens=trace['text_tokens'],generated_token_ids=trace['generated_token_ids'],
            ended_with_eos=trace['ended_with_eos'],model=plan['recipe']['model'],revision=plan['recipe']['revision'],
            prompt=PROMPT,seed=42,max_new_tokens=512,interventions=[],hook_forwards=trace['forwards'])


def append_answers(plan,pair,traces,output,done):
    with (output/'results.jsonl').open('a') as stream:
        for row in answer_rows(plan,pair,traces):
            key=row['question_id'],row['condition']
            if key not in done:stream.write(json.dumps(row)+'\n');done.add(key)


def fit(plan,output):
    import torch
    from analyze_paired_drift import vector_metrics
    totals={};metrics=[];unit_sum=None;valid_count=None;records=[]
    for i,pair in enumerate(plan['pairs']):
        saved=torch.load(output/'pairs'/f'{i:05d}.pt',map_location='cpu',weights_only=True)
        if saved['question_id']!=pair['question_id'] or saved['run_config_sha256']!=digest(output/'run_config.json'):raise ValueError('Pair cache differs from collection plan.')
        n,a=saved['traces']['neutral'],saved['traces']['angry']
        for kind in ('initial','mean'):
            delta=(a[kind]-n[kind]).double();totals[kind]=delta.clone() if kind not in totals else totals[kind]+delta
        _,units,valid,cos,rel,_=vector_metrics(n['initial'],a['initial'],torch)
        unit_sum=units.clone() if unit_sum is None else unit_sum+units
        valid_count=valid.long() if valid_count is None else valid_count+valid
        metrics.append((cos,rel))
        records.extend(answer_rows(plan,pair,saved['traces']))
    for kind,name in [('initial','initial-vectors.pt'),('mean','style-vectors.pt')]:
        sums=totals[kind].float();n=len(plan['pairs'])
        value=dict(protocol=PROTOCOL,run_config_sha256=digest(output/'run_config.json'),kind=kind,groups=GROUPS,
            angry_minus_neutral=sums/n,pair_mean_sums={'angry':sums},pair_counts={'angry':torch.full((len(sums),),n,dtype=torch.long)},
            extraction_question_ids=plan['collection_ids'],evaluation_excluded_ids=plan['evaluation_ids'],prompt=PROMPT,
            model=plan['recipe']['model'],revision=plan['recipe']['revision'],averaging=plan['averaging'] if kind=='mean' else 'Equal-question mean of first-prefill angry-minus-neutral states')
        if (output/name).exists():
            old=torch.load(output/name,map_location='cpu',weights_only=True)
            if old['run_config_sha256']!=value['run_config_sha256'] or not torch.equal(old['angry_minus_neutral'],value['angry_minus_neutral']):raise ValueError('Existing vector differs from frozen pair means.')
        else:save_tensor(torch,value,output/name)
    rows=[]
    for layer in range(len(totals['initial'])):
        for g,group in enumerate(GROUPS):
            rows.append(dict(layer=layer,group=group,pairs=len(metrics),mean_change_pct=mean(float(m[1][layer,g]) for m in metrics),
                median_change_pct=median(float(m[1][layer,g]) for m in metrics),mean_cosine=mean(float(m[0][layer,g]) for m in metrics),
                shift_consistency=float(unit_sum[layer,g].norm()/max(1,int(valid_count[layer,g])))))
    with (output/'layer-drift.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    print('Saved raw collection answers, initial/full-generation vectors and all-layer drift.',flush=True)


def collect(plan,args):
    import torch,transformers
    from step_audio_adapter import make_generator
    if transformers.__version__!=MODELS['step']['transformers']:raise ValueError('Use the existing DIFFA/Step environment.')
    folder=args.output/'pairs';folder.mkdir(exist_ok=True)
    config_hash=digest(args.output/'run_config.json')
    pending=[]
    done={(r['question_id'],r['condition']) for r in read_rows(args.output/'results.jsonl',repair=True)}
    for i,pair in enumerate(plan['pairs']):
        path=folder/f'{i:05d}.pt'
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['question_id']!=pair['question_id'] or saved['run_config_sha256']!=config_hash:raise ValueError('Frozen pair cache differs.')
            append_answers(plan,pair,saved['traces'],args.output,done)
        else:pending.append((i,pair,path))
    if not pending:fit(plan,args.output);return
    # Wait for the entire current Step evaluation, then share the normal GPU lock.
    run_lock=args.evaluation/'run.lock'
    if run_lock.parent.exists():
        with run_lock.open('a') as lock:
            print('Waiting for Step calculation evaluation to finish.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
    with Path('outputs/jenny-task-vectors.lock').open('a') as lock:
        print('Waiting for shared GPU lock.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            print('Waiting for active GPU work.',flush=True);time.sleep(30)
        generator=make_generator(checkpoint('step'))
        (args.output/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,transformers=transformers.__version__,
            model_class=type(generator.model).__name__,layers=len(generator.model.model.layers),
            collector_sha256=digest(Path(__file__)),adapter_sha256=digest(Path(__file__).with_name('step_audio_adapter.py'))),indent=2)+'\n')
        for i,pair,path in pending:
            print(f'Collect [{i+1}/{len(plan["pairs"])}] {pair["question_id"]} neutral + angry',flush=True)
            traces={s:capture(generator,pair[s]) for s in ('neutral','angry')}
            save_tensor(torch,dict(protocol=PROTOCOL,run_config_sha256=config_hash,question_id=pair['question_id'],traces=traces),path)
            append_answers(plan,pair,traces,args.output,done)
            for style,trace in traces.items():print(style+'\n'+trace['response'],flush=True)
    fit(plan,args.output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['prepare','collect','summary'],default='collect')
    p.add_argument('--pairs',type=int,default=400)
    p.add_argument('--evaluation',type=Path,default=Path('outputs/step-gsm400-calculations'))
    p.add_argument('--diffa-source',type=Path,default=Path('outputs/reasoning-transfer/gsm8k'))
    p.add_argument('--audio-sources',type=Path,nargs='+',default=SOURCES)
    p.add_argument('--output',type=Path,default=Path('outputs/step-gsm400-vectors'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='summary':
        import evaluate_gsm_candidates as gsm
        gsm.report(args.output)
    else:
        with (args.output/'queue.lock').open('a') as own:
            fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
            plan=prepare(args)
            if args.mode=='collect':
                collect(plan,args)
                import evaluate_gsm_candidates as gsm
                print('COLLECTION accuracy only; held-out evaluation remains separate.',flush=True);gsm.report(args.output)
