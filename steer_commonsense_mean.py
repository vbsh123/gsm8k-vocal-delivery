"""Learn a CommonsenseQA angry-minus-neutral L31 mean on 250 pairs; test the other 250."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from evaluate_commonsense_audio import PROMPT,score_choice
from extract_audio_vectors import save_tensor,GROUPS
from steer import read_rows,summarize,install_interventions

CONDITIONS=[dict(name='baseline',interventions=[]),dict(name='L31-answer-a1',interventions=[dict(layer=31,group='answer',strength=1)])]


def split_recordings(records,count):
    groups={}
    for row in records:
        if row['style'] in ('neutral','angry'):
            group=groups.setdefault((row['question_id'],row['voice']),{})
            if row['style'] in group:raise ValueError('Duplicate question/style')
            group[row['style']]=row
    keys=sorted(groups)
    if not 0<count<len(keys) or any(set(g)!=set(('neutral','angry')) for g in groups.values()):
        raise ValueError('Need complete neutral/angry pairs and nonempty disjoint extraction/test sets')
    if any(g['neutral']['answer']!=g['angry']['answer'] for g in groups.values()):raise ValueError('Paired references differ')
    random.Random(42).shuffle(keys)
    train=[dict(question_id=q,voice=v,**groups[q,v]) for q,v in keys[:count]]
    test=[groups[q,v]['neutral'] for q,v in keys[count:]]
    return train,test


def report(output):
    print('CommonsenseQA: held-out neutral recordings, L31 answer x1, own angry-minus-neutral mean.',flush=True)
    values=summarize(read_rows(output/'results.jsonl'),CONDITIONS)
    (output/'summary.json').write_text(json.dumps(values,indent=2)+'\n')
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,default=Path('outputs/commonsenseqa-diffa2'))
    p.add_argument('--output',type=Path,default=Path('outputs/commonsenseqa-L31-mean'))
    p.add_argument('--extract-count',type=int,default=250)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.summary_only:report(args.output);return
    source=json.loads((args.benchmark/'run_config.json').read_text())
    if source['prompt']!=PROMPT or source['seed']!=42 or source['max_new_tokens']!=512:
        p.error('Expected the existing seed42 CommonsenseQA DIFFA benchmark')
    train,test=split_recordings(source['recordings'],args.extract_count)
    plan=dict(extraction=train,test=test,seed=42,prompt=PROMPT,layer=31,group='answer',strength=1,
              averaging='mean of matched-forward angry-minus-neutral differences; equal weight per question')
    config=args.output/'split.json'
    if config.exists() and json.loads(config.read_text())!=plan:p.error('Frozen split changed; use a new output')
    config.write_text(json.dumps(plan,indent=2)+'\n')
    print(f'{len(train)} extraction pairs (up to {2*len(train)} generations); {len(test)} disjoint held-out neutral tests. No emotion/outcome filter.',flush=True)
    if args.prepare_only:return
    with (args.output/'queue.lock').open('w') as own:
        try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('CommonsenseQA steering is already running')
        for kind in ('pitch-speed','commonsense'):
            subprocess.run([sys.executable,'-u','wait_for_experiments.py','--kind',kind],check=True)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for the shared GPU lock.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
            run(plan,args.benchmark,args.output)


def run(plan,benchmark,output):
    cache=Path(os.environ.get('HF_HOME') or next((str(p) for p in (Path('/workspace/hf-cache'),Path('/workspace/.hf_home')) if (p/'hub').is_dir()),'outputs/hf-diffa')).resolve()
    os.environ.update(HF_HOME=str(cache),HF_HUB_CACHE=str(cache/'hub'),HUGGINGFACE_HUB_CACHE=str(cache/'hub'))
    os.environ.pop('TRANSFORMERS_CACHE',None)
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from extract_vectors import gen,load_model
    pairs_dir=output/'pairs';pairs_dir.mkdir(exist_ok=True)
    test={r['audio_path']:r for r in plan['test']}
    existing=read_rows(output/'results.jsonl',repair=True)
    if any(r.get('seed')!=42 or r.get('prompt')!=PROMPT for r in existing):raise ValueError('Cached steering recipe differs')
    complete={(r['clip_id'],r['condition']) for r in existing}
    with (output/'results.jsonl').open('a') as stream:
        for row in read_rows(benchmark/'results.jsonl'):
            task=test.get(row['audio_path'])
            if task is None:continue
            if (row['seed']!=42 or row['prompt']!=PROMPT or row['reference_answer']!=task['answer']
                or any(row[k]!=task[k] for k in ('question_id','voice','style'))):
                raise ValueError('Incompatible baseline recipe/reference')
            key=task['audio_path'],'baseline'
            if key not in complete:
                baseline=dict(row,clip_id=task['audio_path'],condition='baseline',reused_from=str(benchmark))
                stream.write(json.dumps(baseline)+'\n');complete.add(key)
    if any((key,'baseline') not in complete for key in test):raise ValueError('Finish unsteered benchmark before steering')
    pending=[(i,pair) for i,pair in enumerate(plan['extraction']) if not (pairs_dir/f'{i:05d}.pt').exists()]
    needed=[task for key,task in test.items() if (key,'L31-answer-a1') not in complete]
    if not pending and not needed:report(output);return
    model,tokenizer,features=load_model();layer=model.llm_model.model.transformer.blocks[31]
    def generate(task):
        file=Path(task['audio_file']);stat=file.stat()
        if stat.st_size!=task['audio_bytes'] or stat.st_mtime_ns!=task['audio_mtime_ns']:raise ValueError('Source recording changed')
        a,sr=sf.read(file,dtype='float32')
        if a.ndim==2:a=a.mean(axis=1)
        if sr!=16000:a=librosa.resample(a,orig_sr=sr,target_sr=16000)
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        return gen(a,16000,model,tokenizer,features,PROMPT,512)
    def capture(task):
        values=[]
        def hook(module,inputs,result):values.append(result[0][0,-512:].float().mean(0).cpu())
        handle=layer.register_forward_hook(hook)
        try:response=generate(task)
        finally:handle.remove()
        return response,torch.stack(values)
    for i,pair in pending:
        print(f'Collect [{i+1}/{len(plan["extraction"])}] {pair["question_id"]} neutral + angry',flush=True)
        nr,n=capture(pair['neutral']);ar,a=capture(pair['angry']);steps=min(len(n),len(a))
        value=dict(question_id=pair['question_id'],difference=(a[:steps]-n[:steps]).mean(0),
                   neutral_response=nr,angry_response=ar,neutral_forwards=len(n),angry_forwards=len(a),matched_forwards=steps)
        save_tensor(torch,value,pairs_dir/f'{i:05d}.pt')
    values=[torch.load(pairs_dir/f'{i:05d}.pt',map_location='cpu',weights_only=True) for i in range(len(plan['extraction']))]
    if any(v['question_id']!=p['question_id'] for v,p in zip(values,plan['extraction'])):raise ValueError('Pair cache differs from split')
    total=torch.stack([v['difference'] for v in values]).sum(0);direction=total/len(values)
    sums=torch.zeros(32,len(GROUPS),len(direction));sums[31,GROUPS.index('answer')]=total
    counts=torch.zeros(32,dtype=torch.long);counts[31]=len(values)
    save_tensor(torch,dict(pair_mean_sums={'angry':sums},pair_counts={'angry':counts},groups=GROUPS,
        prompt=PROMPT,max_new_tokens=512,seed=42,mode='fixed',method='mean',collected_targets=[(31,'answer')],
        extraction_question_ids=[p['question_id'] for p in plan['extraction']],averaging=plan['averaging']),output/'style-vectors.pt')
    directions={(31,'answer'):direction.to(device='cuda',dtype=torch.bfloat16)}
    for index,task in enumerate(needed,1):
        print(f'Steer [{index}/{len(needed)}] {task["question_id"]} L31 answer x1',flush=True)
        with install_interventions(model.llm_model.model.transformer.blocks,directions,CONDITIONS[1]['interventions'],{},512):
            response=generate(task)
        row=dict(audio_path=task['audio_path'],clip_id=task['audio_path'],question_id=task['question_id'],
            voice=task['voice'],style='neutral',condition='L31-answer-a1',seed=42,prompt=PROMPT,response=response,
            **score_choice(response,task['answer']))
        with (output/'results.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(f'  correct={row["correct"]} answer={row["predicted_answer"]}',flush=True)
    report(output)


if __name__=='__main__':main()
