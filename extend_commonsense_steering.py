"""Evaluate the existing L31 vector on new neutral CommonsenseQA recordings."""

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

from evaluate_commonsense_audio import PROMPT, score_choice
from extract_audio_vectors import GROUPS
from steer import install_interventions, read_rows

CONDITION='L31-answer-a1'


def select_records(source, split, exclusions=None):
    previous={r['question_id'] for r in split['test']}|{p['question_id'] for p in split['extraction']}
    records=[]
    for task in read_rows(source/'plan.jsonl'):
        if task['provider']!='azure' or task['style']!='neutral':continue
        if task['question_id'] in previous:raise ValueError('New questions overlap the original extraction/test set')
        file=(source/task['audio_path']).resolve()
        with wave.open(str(file),'rb') as audio:seconds=audio.getnframes()/audio.getframerate()
        if seconds<=0:raise ValueError(f'Empty audio: {file}')
        if seconds>30:
            print(f'Skip >30 seconds, no cropping: {task["question_id"]} ({seconds:.2f}s)',flush=True)
            if exclusions is not None:exclusions.append(dict(question_id=task['question_id'],audio_file=str(file),seconds=seconds))
            continue
        stat=file.stat()
        records.append(dict(task,audio_file=str(file),audio_bytes=stat.st_size,audio_mtime_ns=stat.st_mtime_ns))
    if not records or len({r['question_id'] for r in records})!=len(records):raise ValueError('Need unique neutral questions')
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/azure-commonsenseqa-remaining721'))
    p.add_argument('--original',type=Path,default=Path('outputs/commonsenseqa-angry-mean'))
    p.add_argument('--output',type=Path,default=Path('outputs/commonsenseqa-remaining721-steering'))
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.summary_only:
        from rescore_audio_steering import commonsense_reports
        commonsense_reports(args.original,args.output,args.source,args.output/'rescoring');return
    split=json.loads((args.original/'split.json').read_text())
    vectors=args.original/'style-vectors.pt'
    exclusions=[]
    records=select_records(args.source,split,exclusions)
    recipe=dict(recordings=records,excluded_over_30_seconds=exclusions,vector_sha256=hashlib.sha256(vectors.read_bytes()).hexdigest(),
        vector=str(vectors.resolve()),seed=42,prompt=PROMPT,max_new_tokens=512,decoder='fast_dllm',
        steps=128,block_length=32,temperature=0.,layer=31,group='answer',strength=1.)
    path=args.output/'run_config.json'
    if path.exists() and json.loads(path.read_text())!=recipe:raise ValueError('Frozen inputs/vector changed; use a separate output folder')
    path.write_text(json.dumps(recipe,indent=2)+'\n')
    print(f'{len(recipe["recordings"])} new neutral questions; baseline + existing L31 answer x1. No new extraction.',flush=True)
    if args.prepare_only:return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for kind in ('pitch-speed','commonsense'):
            subprocess.run([sys.executable,'-u','wait_for_experiments.py','--kind',kind],check=True)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for the shared GPU lock.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
            run(recipe,args.output)
    from rescore_audio_steering import commonsense_reports
    commonsense_reports(args.original,args.output,args.source,args.output/'rescoring')


def run(recipe,output):
    cache=Path(os.environ.get('HF_HOME') or next((str(p) for p in
        (Path('/workspace/hf-cache'),Path('/workspace/.hf_home')) if (p/'hub').is_dir()),'outputs/hf-diffa')).resolve()
    os.environ.update(HF_HOME=str(cache),HF_HUB_CACHE=str(cache/'hub'),HUGGINGFACE_HUB_CACHE=str(cache/'hub'))
    os.environ.pop('TRANSFORMERS_CACHE',None)
    import numpy as np
    import soundfile as sf
    import torch
    from extract_vectors import gen,load_model
    saved=read_rows(output/'results.jsonl',repair=True)
    complete={(r['question_id'],r['condition']) for r in saved}
    if all((t['question_id'],c) in complete for t in recipe['recordings'] for c in ('baseline',CONDITION)):return
    state=torch.load(recipe['vector'],map_location='cpu',weights_only=True)
    if state['prompt']!=PROMPT or state['max_new_tokens']!=512:raise ValueError('Vector uses a different task/prompt')
    vector=state['pair_mean_sums']['angry'][31,GROUPS.index('answer')]/state['pair_counts']['angry'][31]
    model,tokenizer,features=load_model();layers=model.llm_model.model.transformer.blocks
    directions={(31,'answer'):vector.to(device='cuda',dtype=torch.bfloat16)}
    for i,task in enumerate(recipe['recordings'],1):
        if all((task['question_id'],c) in complete for c in ('baseline',CONDITION)):continue
        audio,sr=sf.read(task['audio_file'],dtype='float32')
        for condition in ('baseline',CONDITION):
            if (task['question_id'],condition) in complete:continue
            random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            print(f'[{i}/{len(recipe["recordings"])}] {task["question_id"]} {condition}',flush=True)
            interventions=[] if condition=='baseline' else [dict(layer=31,group='answer',strength=1)]
            with install_interventions(layers,directions,interventions,{},512):
                response=gen(audio,sr,model,tokenizer,features,PROMPT,512)
            row=dict(question_id=task['question_id'],voice=task['voice'],style='neutral',clip_id=task['audio_path'],
                audio_path=task['audio_path'],condition=condition,response=response,seed=42,prompt=PROMPT,
                max_new_tokens=512,decoder='fast_dllm',**score_choice(response,task['answer']))
            with (output/'results.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print(f'  correct={row["correct"]} answer={row["predicted_answer"]}',flush=True)


if __name__=='__main__':main()
