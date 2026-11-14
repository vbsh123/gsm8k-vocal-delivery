"""Learn an ARC L31 angry-minus-neutral mean on 250 pairs; test on the disjoint remaining neutrals."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from evaluate_arc_audio import PROMPT,STEERING,report,score
from extract_audio_vectors import GROUPS,save_tensor
from steer import install_interventions,read_rows

ARC_STEERING='neutral-ARC-L31-answer-a1'
CS_STEERING='neutral-CS-L31-answer-a1'


def split_recordings(records,count=250):
    neutral={};angry={}
    for row in records:
        if row['condition'] not in ('neutral','angry'):continue
        target=neutral if row['condition']=='neutral' else angry
        key=row['question_id'],row['voice']
        if key in target:raise ValueError('Duplicate question/voice in ARC source')
        target[key]=row
    keys=sorted(neutral.keys()&angry.keys());random.Random(42).shuffle(keys)
    if count<1 or len(keys)<count or len(neutral)<=count:raise ValueError('Need enough complete training pairs and disjoint neutral test questions')
    selected=set(keys[:count])
    train=[dict(question_id=q,voice=v,neutral=neutral[q,v],angry=angry[q,v]) for q,v in keys[:count]]
    test=[r for key,r in sorted(neutral.items()) if key not in selected]
    return train,test


def summary(output,source,benchmark):
    plan=json.loads((output/'split.json').read_text())
    train={r['question_id'] for r in plan['extraction']}
    keys={(r['question_id'],r['voice']) for r in plan['test']}
    rows=read_rows(output/'results.jsonl')
    if train&{r['question_id'] for r in rows}:raise ValueError('Training questions found in ARC-vector evaluation')
    print(f'\nARC-trained vector: {len(plan["extraction"])} extraction pairs; {len(plan["test"])} disjoint neutral tests.',flush=True)
    report(output,source)
    transfer=[dict(r,condition=CS_STEERING) for r in read_rows(benchmark/'results.jsonl')
        if r['condition']==STEERING and (r['question_id'],r['voice']) in keys]
    common={(r['question_id'],r['voice']) for r in rows if r['condition']==ARC_STEERING}
    common&={(r['question_id'],r['voice']) for r in transfer}
    comparison=[r for r in rows+transfer if (r['question_id'],r['voice']) in common]
    print(f'\nCOMMON HELD-OUT COHORT: both vectors completed on {len(common)} questions; same neutral baselines.',flush=True)
    report(output,source,rows=comparison,summary_name='shared-heldout-summary.json')
    have={(r['question_id'],r['voice'],r['condition']) for r in rows}
    pending=sum((q,v,ARC_STEERING) not in have for q,v in keys)
    print(f'ARC-vector interventions pending: {pending}/{len(keys)}. Full-500 CommonsenseQA transfer summary remains in {benchmark}/summary.json.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/azure-arc-challenge500'))
    p.add_argument('--benchmark',type=Path,default=Path('outputs/arc-challenge-diffa2'))
    p.add_argument('--output',type=Path,default=Path('outputs/arc-challenge-angry-mean'))
    p.add_argument('--extract-count',type=int,default=250)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.summary_only:summary(args.output,args.source,args.benchmark);return
    config=json.loads((args.benchmark/'run_config.json').read_text())
    if config['prompt']!=PROMPT or config['seed']!=42 or config['max_new_tokens']!=512 or config['decoder']!='fast_dllm':
        raise ValueError('Expected the existing seed42 ARC fast_dllm benchmark')
    train,test=split_recordings(config['recordings'],args.extract_count)
    plan=dict(extraction=train,test=test,seed=42,prompt=PROMPT,layer=31,group='answer',strength=1,
        averaging='Matched forward differences, then equal weight per extraction question')
    path=args.output/'split.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Frozen ARC split changed; use a new output folder')
    path.write_text(json.dumps(plan,indent=2)+'\n')
    print(f'{len(train)} fixed ARC extraction pairs; {len(test)} disjoint neutral tests; L31 answer x1, seed42.',flush=True)
    if args.prepare_only:return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock; the ARC delivery/transfer evaluation finishes first.',flush=True)
            fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
            run(plan,args.source,args.benchmark,args.output)
    summary(args.output,args.source,args.benchmark)


def run(plan,source,benchmark,output):
    bank={r['question_id']:r for r in read_rows(source/'questions.jsonl')}
    cached={(r['question_id'],r['voice'],r['condition']):r for r in read_rows(benchmark/'results.jsonl')}
    saved=read_rows(output/'results.jsonl',repair=True)
    complete={(r['question_id'],r['voice'],r['condition']) for r in saved}
    with (output/'results.jsonl').open('a') as stream:
        for task in plan['test']:
            key=task['question_id'],task['voice'],'neutral'
            baseline=cached.get(key)
            if baseline is None:raise ValueError('Finish the ARC unsteered benchmark before collecting/evaluating')
            if baseline['seed']!=42 or baseline['prompt']!=PROMPT or baseline['reference_answer']!=task['answer']:
                raise ValueError('Incompatible ARC baseline')
            if key not in complete:
                stream.write(json.dumps(dict(baseline,reused_from=str(benchmark)))+'\n');complete.add(key)
    pairs=output/'pairs';pairs.mkdir(exist_ok=True)
    pending=[(i,p) for i,p in enumerate(plan['extraction']) if not (pairs/f'{i:05d}.pt').exists()]
    needed=[t for t in plan['test'] if (t['question_id'],t['voice'],ARC_STEERING) not in complete]
    if not pending and not needed:return
    cache=Path(os.environ.get('HF_HOME') or next((str(p) for p in
        (Path('/workspace/hf-cache'),Path('/workspace/.hf_home')) if (p/'hub').is_dir()),'outputs/hf-diffa')).resolve()
    os.environ.update(HF_HOME=str(cache),HF_HUB_CACHE=str(cache/'hub'),HUGGINGFACE_HUB_CACHE=str(cache/'hub'))
    os.environ.pop('TRANSFORMERS_CACHE',None)
    import numpy as np
    import soundfile as sf
    import torch
    from extract_vectors import gen,load_model
    model,tokenizer,features=load_model();layers=model.llm_model.model.transformer.blocks
    def generate(task):
        file=Path(task['audio_file']);stat=file.stat()
        if stat.st_size!=task['audio_bytes'] or stat.st_mtime_ns!=task['audio_mtime_ns']:raise ValueError('ARC audio changed')
        audio,sr=sf.read(file,dtype='float32')
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        return gen(audio,sr,model,tokenizer,features,PROMPT,512)
    def capture(task):
        values=[]
        def hook(module,inputs,result):values.append(result[0][0,-512:].float().mean(0).cpu())
        handle=layers[31].register_forward_hook(hook)
        try:response=generate(task)
        finally:handle.remove()
        return response,torch.stack(values)
    for i,pair in pending:
        print(f'Collect [{i+1}/{len(plan["extraction"])}] {pair["question_id"]} neutral + angry',flush=True)
        nr,n=capture(pair['neutral']);ar,a=capture(pair['angry']);steps=min(len(n),len(a))
        if i<3:
            for style,response in [('neutral',nr),('angry',ar)]:
                old=cached.get((pair['question_id'],pair['voice'],style))
                if old is not None and old['response']!=response:raise ValueError('Non-mutating collection changed a benchmark response; inspect model/decoder parity')
        save_tensor(torch,dict(question_id=pair['question_id'],difference=(a[:steps]-n[:steps]).mean(0),
            neutral_response=nr,angry_response=ar,neutral_forwards=len(n),angry_forwards=len(a),matched_forwards=steps),pairs/f'{i:05d}.pt')
    values=[torch.load(pairs/f'{i:05d}.pt',map_location='cpu',weights_only=True) for i in range(len(plan['extraction']))]
    if any(v['question_id']!=p['question_id'] for v,p in zip(values,plan['extraction'])):raise ValueError('Pair cache differs from split')
    total=torch.stack([v['difference'] for v in values]).sum(0);direction=total/len(values)
    sums=torch.zeros(32,len(GROUPS),len(direction));sums[31,GROUPS.index('answer')]=total
    counts=torch.zeros(32,dtype=torch.long);counts[31]=len(values)
    save_tensor(torch,dict(pair_mean_sums={'angry':sums},pair_counts={'angry':counts},groups=GROUPS,
        prompt=PROMPT,max_new_tokens=512,seed=42,mode='fixed',method='mean',collected_targets=[(31,'answer')],
        extraction_question_ids=[p['question_id'] for p in plan['extraction']],averaging=plan['averaging']),output/'style-vectors.pt')
    directions={(31,'answer'):direction.to(device='cuda',dtype=torch.bfloat16)}
    for i,task in enumerate(needed,1):
        print(f'ARC steering [{i}/{len(needed)}] {task["question_id"]} L31 answer x1',flush=True)
        with install_interventions(layers,directions,[dict(layer=31,group='answer',strength=1)],{},512):
            response=generate(task)
        row=dict(question_id=task['question_id'],voice=task['voice'],style='neutral',audio_path=task['audio_path'],
            condition=ARC_STEERING,seed=42,prompt=PROMPT,max_new_tokens=512,decoder='fast_dllm',response=response,
            **score(response,bank[task['question_id']]))
        with (output/'results.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(f'  answer_correct={row["answer_correct"]}; format_compliant={row["format_compliant"]}',flush=True)


if __name__=='__main__':main()
