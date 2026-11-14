"""Evaluate ARC deliveries and optionally a frozen L31 vector; report answer accuracy and format separately."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import wave

from audit_commonsense_scoring import RELAXED_SCORER_VERSION,relaxed_choice
from evaluate_commonsense_audio import score_choice
from rescore_audio_steering import paired
from steer import install_interventions,read_rows

PROMPT=('Answer the multiple-choice question in the audio. The audio includes the question and '
    'four choices labeled A, B, C, D. Give concise reasoning. End with one plain-text line '
    'in exactly this format: Final answer: <letter>. Give only one letter from A to D on that line.')
STEERING='neutral-L31-answer-a1'


def score(response,question):
    strict=score_choice(response,question['answer'])
    prediction=relaxed_choice(response,question['choices'])
    labels={c['label'] for c in question['choices']}
    if prediction not in labels:prediction=None
    return dict(**strict,answer_correct=prediction==question['answer'],answer_prediction=prediction,
        answer_parse_failed=prediction is None,format_compliant=strict['predicted_answer'] in labels,
        answer_scorer_version=RELAXED_SCORER_VERSION)


def report(output,source,rows=None,summary_name='summary.json'):
    bank={r['question_id']:r for r in read_rows(source/'questions.jsonl')}
    rows=read_rows(output/'results.jsonl') if rows is None else rows
    latest={(r['question_id'],r['voice'],r['condition']):r for r in rows}
    groups={}
    for (q,voice,condition),row in latest.items():
        if row['reference_answer']!=bank[q]['answer']:raise ValueError('Saved reference differs from ARC source')
        groups.setdefault(condition,{})[q,voice]=score(row['response'],bank[q])
    print('\nARC-Challenge: custom answer scorer '+RELAXED_SCORER_VERSION+'; original generations unchanged.')
    print(f'{"Condition":30} {"N":>5} {"Answer accuracy":>16} {"Format compliant":>17} {"Strict accuracy":>16}')
    accuracy=[];comparisons=[]
    for condition,values in sorted(groups.items()):
        n=len(values);a=sum(r['answer_correct'] for r in values.values())
        f=sum(r['format_compliant'] for r in values.values());s=sum(r['correct'] for r in values.values())
        print(f'{condition:30} {n:5} {a/n:16.1%} {f/n:17.1%} {s/n:16.1%}')
        accuracy.append(dict(condition=condition,n=n,answer_correct=a,format_compliant=f,strict_correct=s))
    base=groups.get('neutral',{})
    print('\nPaired against the SAME neutral question/voice; answer correctness first, strict accuracy second.')
    for condition,values in sorted(groups.items()):
        if condition=='neutral':continue
        keys=base.keys()&values.keys()
        if not keys:continue
        for metric in ('answer_correct','correct'):
            result=paired([(base[k][metric],values[k][metric]) for k in sorted(keys)])
            lo,hi=result['ci95_gain_pp'];n=result['pairs']
            print(f'{condition} / {metric}: N={n}; gain={result["gain_pp"]:+.2f} pp; '
                f'wins/losses={result["wins"]}/{result["losses"]}; p={result["mcnemar_exact_p"]:.4f}; CI=[{lo:+.1f}, {hi:+.1f}] pp')
            comparisons.append(dict(condition=condition,metric=metric,**result))
    (output/summary_name).write_text(json.dumps(dict(answer_scorer=RELAXED_SCORER_VERSION,accuracy=accuracy,paired=comparisons),indent=2)+'\n')
    print('P-values and 95% paired bootstrap intervals are unadjusted; parse failures count as incorrect.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/azure-arc-challenge500'))
    p.add_argument('--output',type=Path,default=Path('outputs/arc-challenge-diffa2'))
    p.add_argument('--vectors',type=Path,help='Existing angry-minus-neutral mean; apply L31 answer x1 without fitting on ARC')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.summary_only:report(args.output,args.source);return
    bank={r['question_id']:r for r in read_rows(args.source/'questions.jsonl')}
    tasks=[];excluded=[]
    for row in read_rows(args.source/'plan.jsonl'):
        if row['provider']!='azure':continue
        q=bank[row['question_id']]
        if row['answer']!=q['answer'] or row['question']!=q['question']:raise ValueError('Audio plan differs from question bank')
        path=(args.source/row['audio_path']).resolve()
        with wave.open(str(path),'rb') as audio:seconds=audio.getnframes()/audio.getframerate()
        if seconds<=0:raise ValueError('Empty ARC recording')
        if seconds>30:
            excluded.append(dict(question_id=row['question_id'],style=row['style'],seconds=seconds));continue
        stat=path.stat()
        task=dict(row,audio_file=str(path),audio_bytes=stat.st_size,audio_mtime_ns=stat.st_mtime_ns,condition=row['style'])
        tasks.append(task)
        if args.vectors and row['style']=='neutral':tasks.append(dict(task,condition=STEERING))
    if not tasks:raise ValueError('No eligible ARC audio')
    vector=None
    if args.vectors:
        vector=dict(path=str(args.vectors.resolve()),sha256=hashlib.sha256(args.vectors.read_bytes()).hexdigest())
    config=dict(recordings=tasks,excluded_over_30_seconds=excluded,vector=vector,prompt=PROMPT,seed=42,
        decoder='fast_dllm',steps=128,block_length=32,max_new_tokens=512,temperature=0.)
    path=args.output/'run_config.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Frozen ARC inputs/settings changed; use a new output folder')
    path.write_text(json.dumps(config,indent=2)+'\n')
    print(f'{len(tasks)} planned answers; {len(excluded)} recordings over 30s excluded without cropping.',flush=True)
    if args.prepare_only:return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for kind in ('pitch-speed','audio-models','commonsense'):
            subprocess.run([sys.executable,'-u','wait_for_experiments.py','--kind',kind],check=True)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
            run(config,bank,args.output)
    report(args.output,args.source)


def run(config,bank,output):
    saved=read_rows(output/'results.jsonl',repair=True)
    complete={(r['question_id'],r['voice'],r['condition']) for r in saved}
    pending=[t for t in config['recordings'] if (t['question_id'],t['voice'],t['condition']) not in complete]
    if not pending:return
    cache=Path(os.environ.get('HF_HOME') or next((str(p) for p in (Path('/workspace/hf-cache'),Path('/workspace/.hf_home')) if (p/'hub').is_dir()),'outputs/hf-diffa')).resolve()
    os.environ.update(HF_HOME=str(cache),HF_HUB_CACHE=str(cache/'hub'),HUGGINGFACE_HUB_CACHE=str(cache/'hub'))
    os.environ.pop('TRANSFORMERS_CACHE',None)
    import numpy as np
    import soundfile as sf
    import torch
    from extract_vectors import gen,load_model
    directions={}
    if config['vector']:
        state=torch.load(config['vector']['path'],map_location='cpu',weights_only=True)
        training=set(state.get('extraction_question_ids',[]))
        if training&set(bank):raise ValueError('ARC questions overlap vector extraction')
        index=state['groups'].index('answer');count=state['pair_counts']['angry'][31]
        if count<=0:raise ValueError('No L31 mean in this vector')
        directions[31,'answer']=(state['pair_mean_sums']['angry'][31,index]/count).to(device='cuda',dtype=torch.bfloat16)
    model,tokenizer,features=load_model();layers=model.llm_model.model.transformer.blocks
    for i,task in enumerate(pending,1):
        stat=Path(task['audio_file']).stat()
        if stat.st_size!=task['audio_bytes'] or stat.st_mtime_ns!=task['audio_mtime_ns']:raise ValueError('ARC audio changed')
        audio,sr=sf.read(task['audio_file'],dtype='float32')
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        interventions=[dict(layer=31,group='answer',strength=1)] if task['condition']==STEERING else []
        print(f'[{i}/{len(pending)}] {task["question_id"]} {task["condition"]}',flush=True)
        started=time.monotonic()
        with install_interventions(layers,directions,interventions,{},512):
            response=gen(audio,sr,model,tokenizer,features,PROMPT,512)
        row=dict(question_id=task['question_id'],voice=task['voice'],style=task['style'],audio_path=task['audio_path'],
            condition=task['condition'],seed=42,prompt=PROMPT,max_new_tokens=512,decoder='fast_dllm',
            response=response,generation_seconds=time.monotonic()-started,**score(response,bank[task['question_id']]))
        with (output/'results.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(f'  answer_correct={row["answer_correct"]}; format_compliant={row["format_compliant"]}',flush=True)


if __name__=='__main__':main()
