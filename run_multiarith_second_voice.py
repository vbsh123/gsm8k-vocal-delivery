"""DIFFA MultiArith: same290 evaluation questions, Azure Guy neutral versus angry."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import run_reasoning_transfer as reasoning
from steer import read_rows
from winning_diffa_sweeps import neutral_records


def prepare(args):
    original_path=args.original/'run_config.json'; original=json.loads(original_path.read_text())
    if (original['task']!='multiarith' or original['decoder']!='fast_dllm' or original['seed']!=42
            or original['max_new_tokens']!=512 or original.get('steps',128)!=128
            or original.get('block_length',32)!=32 or original.get('temperature',0)!=0):
        raise ValueError('Need original frozen MultiArith DIFFA evaluation recipe.')
    ids=[r['question_id'] for r in neutral_records(original)]
    if len(ids)!=290 or len(set(ids))!=290:raise ValueError('Expected same290 evaluation questions.')
    questions=[original['bank'][q] for q in ids]
    args.source.mkdir(parents=True,exist_ok=True);args.output.mkdir(parents=True,exist_ok=True)
    selection=dict(task='multiarith',source_run=str(args.original.absolute()),source_config_sha256=reasoning.digest(original_path),
        question_ids=ids,voice=args.voice,styles=['neutral','angry'],intensity=1.5,prompt=original['prompt'],
        rule='Same frozen290 evaluation questions, new voice, no outcome/classifier filtering; no collection or steering.')
    reasoning.freeze(args.output/'selection.json',selection)
    reasoning.bank_file(args.source/'questions.jsonl',questions)
    subprocess.run([sys.executable,'-u','cloud_tts_smoke.py','--provider','azure','--azure-voice',args.voice,
        '--azure-intensity','1.5','--input-jsonl',str(args.source/'questions.jsonl'),'--num-examples','290',
        '--styles','neutral','angry','--output',str(args.source),'--resume','--retry-passes','5'],check=True)
    records=reasoning.source_records(args.source,{q['question_id']:q for q in questions},['neutral','angry'])
    tasks=[dict(records[q,s],condition=s) for q in ids for s in ('neutral','angry')]
    plan=dict(task='multiarith',protocol='diffa-multiarith290-second-voice-delivery-v1',bank={q['question_id']:q for q in questions},
        recordings=tasks,prompt=original['prompt'],seed=42,max_new_tokens=512,decoder='fast_dllm',steps=128,block_length=32,
        temperature=0.,voice=args.voice,source_config_sha256=selection['source_config_sha256'],report_label='Unsteered neutral/angry delivery',
        selection=selection['rule'],scoring_note='Installed upstream GSM8K task filters/metric on MultiArith references; '
            'numeric equivalence additional. New voice delivery comparison only; no steering. Native DIFFA audio path '
            'up to30s and existing chunked-Whisper/global-adapter path beyond30s; no cropping.')
    reasoning.freeze(args.output/'run_config.json',plan)
    print('New voice:',args.voice,';290 matched questions,580 unsteered answers.',flush=True)
    return plan


def evaluate(plan,output):
    import numpy as np
    import soundfile as sf
    import torch
    from extract_vectors import load_model,gen_chunked
    tasks={(r['question_id'],r['voice'],r['condition']):r for r in plan['recordings']};done=set()
    for row in read_rows(output/'results.jsonl',repair=True):
        key=row['question_id'],row['voice'],row['condition']
        if (key not in tasks or key in done or row['prompt']!=plan['prompt'] or row['seed']!=42
                or row['decoder']!='fast_dllm' or row['audio_sha256']!=tasks[key]['audio_sha256']
                or row['max_new_tokens']!=512 or row['interventions']):raise ValueError('Saved new-voice response differs.')
        done.add(key)
    jobs=[(key,r) for key,r in tasks.items() if key not in done]
    if not jobs:print('All580 new-voice answers saved; no model loaded.');return
    cache=Path(os.environ.get('HF_HOME','/workspace/.hf_home')).absolute()
    os.environ.update(HF_HOME=str(cache),HF_HUB_CACHE=str(cache/'hub'),HUGGINGFACE_HUB_CACHE=str(cache/'hub'))
    os.environ.pop('TRANSFORMERS_CACHE',None)
    model,tokenizer,features=load_model()
    (output/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        loader_sha256=reasoning.digest(Path('extract_vectors.py')),script_sha256=reasoning.digest(__file__),
        generation='DIFFA BF16 fast_dllm;128 steps,block32,512 tokens,temperature0,seed42;no hooks'),indent=2)+'\n')
    for i,(key,task) in enumerate(jobs,1):
        if reasoning.digest(task['audio_file'])!=task['audio_sha256']:raise ValueError('Audio changed.')
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        audio,rate=sf.read(task['audio_file'],dtype='float32')
        print(f'New voice [{i}/{len(jobs)}] {task["question_id"]} {task["condition"]}',flush=True)
        response,encoding=gen_chunked(audio,rate,model,tokenizer,features,plan['prompt'],512,30)
        row=dict(task,reference_answer=task['answer'],response=response,audio_encoding=encoding,prompt=plan['prompt'],
            seed=42,max_new_tokens=512,decoder='fast_dllm',steps=128,block_length=32,temperature=0.,interventions=[])
        with (output/'results.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(response,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','evaluate','summary'),default='evaluate')
    p.add_argument('--original',type=Path,default=Path('outputs/multiarith580-angry-mean'))
    p.add_argument('--source',type=Path,default=Path('outputs/azure-multiarith290-guy'))
    p.add_argument('--output',type=Path,default=Path('outputs/multiarith290-guy-delivery'))
    p.add_argument('--voice',default='en-US-GuyNeural')
    args=p.parse_args()
    if args.mode=='summary':reasoning.report(args.output);return
    if args.mode=='prepare':prepare(args);return
    plan=json.loads((args.output/'run_config.json').read_text())
    import fcntl
    with Path('outputs/jenny-task-vectors.lock').open('w') as gpu:
        print('Waiting for shared DIFFA GPU lock.',flush=True);fcntl.flock(gpu,fcntl.LOCK_EX)
        import time
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            print('Waiting for active GPU work.',flush=True);time.sleep(30)
        evaluate(plan,args.output)


if __name__=='__main__':main()
