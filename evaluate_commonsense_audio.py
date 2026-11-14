"""DIFFA evaluation of spoken CommonsenseQA; A–E scoring and matched delivery comparisons."""

import argparse
import fcntl
from itertools import combinations
import json
import os
from math import comb
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import wave

from steer import read_rows

STYLES=['neutral','angry','happy','sad']
PROMPT=('Answer the multiple-choice question in the audio. The audio includes the question and '
        'five choices labeled A, B, C, D, E. Give concise reasoning. End with one plain-text line '
        'in exactly this format: Final answer: <letter>. Give only one letter from A to E on that line.')


def score_choice(response,reference):
    clean=response.replace('**','').replace('__','').replace('`','')
    labels=list(re.finditer(r'final\s+answer\s*:',clean,re.I))
    tail=clean[labels[-1].end():].strip().splitlines() if labels else []
    match=re.fullmatch(r'(?:option\s+)?\(?([A-E])\)?[.!]?',tail[0].strip(),re.I) if tail else None
    predicted=match[1].upper() if match else None
    if reference not in list('ABCDE'):raise ValueError('Expected an A–E reference')
    return dict(predicted_answer=predicted,reference_answer=reference,parse_failed=predicted is None,
                correct=predicted==reference,scorer_version='explicit-choice-final-v1')


def report(output,rows=None,summary_name='summary.json'):
    import numpy as np
    rows=read_rows(output/'results.jsonl') if rows is None else rows
    latest={r['audio_path']:r for r in rows}
    groups={s:{} for s in STYLES}
    print('\nCommonsenseQA: unfiltered delivery accuracy; parse failures count as incorrect.')
    for row in latest.values():
        value=score_choice(row['response'],row['reference_answer'])['correct']
        groups[row['style']][row['question_id'],row['voice']]=value
    for style,values in groups.items():
        print(f'{style:8}: {sum(values.values())}/{len(values)} correct'
              +(f' ({sum(values.values())/len(values):.1%})' if values else ''))
    summary=[]
    print(f'\n{"Reference":9} {"Test":8} {"Pairs":>5} {"Ref acc":>8} {"Test acc":>8} {"Gain pp":>8} {"Wins":>5} {"Losses":>6} {"Exact p":>9}   95% CI gain pp')
    for a,b in combinations(STYLES,2):
        keys=groups[a].keys()&groups[b].keys();n=len(keys)
        if not n:continue
        left,right=groups[a],groups[b]
        w=sum(right[k] and not left[k] for k in keys);l=sum(left[k] and not right[k] for k in keys)
        d=w+l;p=min(1.,2*sum(comb(d,i) for i in range(min(w,l)+1))/2**d)
        x=sum(left[k] for k in keys);y=sum(right[k] for k in keys)
        draws=np.random.default_rng(42).multinomial(n,[w/n,l/n,(n-d)/n],size=200000)
        ci=np.percentile(100*(draws[:,0]-draws[:,1])/n,[2.5,97.5]).tolist()
        summary.append(dict(reference=a,test=b,pairs=n,ref_correct=x,test_correct=y,wins=w,losses=l,mcnemar_exact_p=p,ci95_gain_pp=ci))
        print(f'{a:9} {b:8} {n:5} {x/n:8.1%} {y/n:8.1%} {100*(y-x)/n:+8.2f} {w:5} {l:6} {p:9.4f}   [{ci[0]:+.1f}, {ci[1]:+.1f}]')
    print('Same question and voice; paired bootstrap intervals and McNemar p-values are unadjusted.')
    (output/summary_name).write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/azure-commonsenseqa500'))
    p.add_argument('--output',type=Path,default=Path('outputs/commonsenseqa-diffa2'))
    p.add_argument('--wait-for-speeds',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.summary_only:report(args.output);return
    tasks=[r for r in read_rows(args.source/'plan.jsonl') if r['provider']=='azure' and r['style'] in STYLES]
    if not tasks:raise ValueError('No CommonsenseQA recordings found')
    seen=set();groups={};records=[]
    for task in tasks:
        key=task['question_id'],task['voice'],task['style']
        if key in seen:raise ValueError('Duplicate question/voice/style')
        seen.add(key);groups.setdefault(key[:2],set()).add(task['style'])
        file=(args.source/task['audio_path']).resolve();stat=file.stat()
        with wave.open(str(file),'rb') as audio:duration=audio.getnframes()/audio.getframerate()
        if not 0<duration<=30:raise ValueError(f'Recording exceeds 30s or is empty: {file}; no cropping')
        if task['answer'] not in list('ABCDE'):raise ValueError('Wrong dataset: expected A–E answers')
        records.append(dict(task,audio_file=str(file),audio_bytes=stat.st_size,audio_mtime_ns=stat.st_mtime_ns))
    if any(styles!=set(STYLES) for styles in groups.values()):raise ValueError('Every question must have all four deliveries')
    config=dict(recordings=records,seed=42,prompt=PROMPT,max_new_tokens=512,decoder='fast_dllm',
                steps=128,block_length=32,temperature=0.,scorer_version='explicit-choice-final-v1')
    path=args.output/'run_config.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Inputs or settings changed; use a different output folder')
    path.write_text(json.dumps(config,indent=2)+'\n')
    saved=read_rows(args.output/'results.jsonl',repair=not args.prepare_only)
    if any(r.get('seed')!=42 or r.get('prompt')!=PROMPT for r in saved):raise ValueError('Cached answers use different settings')
    complete={r['audio_path'] for r in saved}
    pending=[r for r in records if r['audio_path'] not in complete]
    print(f'{len(groups)} matched questions x 4 deliveries; {len(pending)}/{len(records)} answers pending. No emotion or correctness filtering.',flush=True)
    if args.prepare_only:return
    if not pending:report(args.output);return
    with (args.output/'queue.lock').open('w') as own:
        try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('CommonsenseQA evaluation is already queued or running')
        if args.wait_for_speeds:
            subprocess.run([sys.executable,'-u','wait_for_experiments.py','--kind','pitch-speed'],check=True)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for the shared GPU lock.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
            cache=Path(os.environ.get('HF_HOME') or next((str(p) for p in
                (Path('/workspace/hf-cache'),Path('/workspace/.hf_home')) if (p/'hub').is_dir()),'outputs/hf-diffa')).resolve()
            os.environ['HF_HOME']=str(cache)
            os.environ['HF_HUB_CACHE']=str(cache/'hub')
            os.environ['HUGGINGFACE_HUB_CACHE']=str(cache/'hub')
            os.environ.pop('TRANSFORMERS_CACHE',None)
            import librosa
            import numpy as np
            import soundfile as sf
            import torch
            from extract_vectors import gen,load_model
            model,tokenizer,features=load_model()
            for index,task in enumerate(pending,1):
                audio,sr=sf.read(task['audio_file'],dtype='float32')
                if audio.ndim==2:audio=audio.mean(axis=1)
                if sr!=16000:audio=librosa.resample(audio,orig_sr=sr,target_sr=16000)
                random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
                print(f'[{index}/{len(pending)}] {task["question_id"]} {task["style"]}',flush=True)
                started=time.monotonic();response=gen(audio,16000,model,tokenizer,features,PROMPT,512)
                row={k:task[k] for k in ('audio_path','question_id','voice','style')}
                row.update(seed=42,prompt=PROMPT,max_new_tokens=512,decoder='fast_dllm',response=response,
                           generation_seconds=time.monotonic()-started,**score_choice(response,task['answer']))
                with (args.output/'results.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                print(f'  correct={row["correct"]} answer={row["predicted_answer"]}\n{response}',flush=True)
            report(args.output)


if __name__=='__main__':main()
