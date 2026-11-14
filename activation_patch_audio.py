"""Matched-state angry-to-neutral activation patching; diagnostic, requires both recordings."""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import wave

from compare_cloud_accuracy import PROMPT
from scoring import score_response
from steer import read_rows,summarize

TARGETS=[('patch-L12-acoustic',12,'acoustic'),('patch-L12-answer',12,'answer'),('patch-L31-answer-control',31,'answer')]
CONDITIONS=[dict(name=n) for n in ('baseline','angry-unpatched',*[t[0] for t in TARGETS])]
ANSWER_TOKENS=512


def positions(group,bounds):
    return slice(-ANSWER_TOKENS,None) if group=='answer' else slice(bounds['start'],bounds['acoustic_end'])


@contextmanager
def patch_forward(llm,layer,donor_prefix,donor_bounds,recipient_bounds,group,torch,
                  collect_only=False,max_forwards=None):
    state=dict(donor=False,recipient_forwards=0,donor_forwards=0,patched_forwards=0)
    if collect_only:state.update(neutral_means=[],differences=[])
    captured={}

    def before(module,args,kwargs):
        if state['donor']:return
        if args or set(kwargs)!={'inputs_embeds'}:
            raise ValueError('Expected pinned fast-dllm inputs_embeds-only forward')
        embeds=kwargs['inputs_embeds']
        if embeds.shape[0]!=1 or embeds.shape[1]<=ANSWER_TOKENS:
            raise ValueError('Expected one full prompt and 512 answer embeddings')
        state['recipient_forwards']+=1
        if collect_only and max_forwards is not None and state['recipient_forwards']>max_forwards:
            captured.clear();return
        # Reuse current recipient answer embeddings, including all committed tokens and masks.
        donor=torch.cat([donor_prefix,embeds[:,-ANSWER_TOKENS:]],dim=1)
        captured.clear();state['donor']=True
        try:llm(inputs_embeds=donor)
        finally:state['donor']=False
        if 'hidden' not in captured:raise RuntimeError('Donor layer did not run')
        state['donor_forwards']+=1

    def after(module,args,result):
        if state['donor']:
            captured['hidden']=result[0][:,positions(group,donor_bounds)].detach().clone()
            return
        index=positions(group,recipient_bounds)
        if collect_only and 'hidden' not in captured:return
        if result[0][:,index].shape!=captured['hidden'].shape:
            raise ValueError('Donor and recipient token groups are not aligned')
        if collect_only:
            neutral=result[0][:,index].float();angry=captured.pop('hidden').float()
            state['neutral_means'].append(neutral.mean(dim=(0,1)).cpu())
            state['differences'].append((angry-neutral).mean(dim=(0,1)).cpu())
            return
        hidden=result[0].clone()
        hidden[:,index]=captured.pop('hidden')
        state['patched_forwards']+=1
        return (hidden,*result[1:])

    handles=[llm.register_forward_pre_hook(before,with_kwargs=True),layer.register_forward_hook(after)]
    try:yield state
    finally:
        for handle in handles:handle.remove()


class PrefixCaptured(Exception):
    pass


def capture_prefix(llm,generate,bounds):
    saved={}
    def before(module,args,kwargs):
        if args or set(kwargs)!={'inputs_embeds'}:raise ValueError('Unexpected decoder signature')
        saved['prefix']=kwargs['inputs_embeds'][:,:-ANSWER_TOKENS].detach().clone()
        saved['bounds']=dict(bounds)
        raise PrefixCaptured()
    handle=llm.register_forward_pre_hook(before,with_kwargs=True)
    try:
        try:generate()
        except PrefixCaptured:pass
    finally:handle.remove()
    if 'prefix' not in saved:raise RuntimeError('Decoder input was not captured')
    return saved['prefix'],saved['bounds']


def select_pairs(paths,count):
    groups={}
    for folder in paths:
        for task in read_rows(folder/'plan.jsonl'):
            if task['provider']!='azure' or task['style'] not in ('neutral','angry'):continue
            key=task['question_id'],task['voice'];style=task['style']
            file=(folder/task['audio_path']).resolve()
            with wave.open(str(file),'rb') as audio:duration=audio.getnframes()/audio.getframerate()
            if not 0<duration<=30:continue
            if style in groups.setdefault(key,{}):raise ValueError('Duplicate recording')
            stat=file.stat()
            groups[key][style]=dict(task,audio_file=str(file),source_folder=str(folder.resolve()),
                                    audio_bytes=stat.st_size,audio_mtime_ns=stat.st_mtime_ns)
    pairs=[]
    for (qid,voice),clips in sorted(groups.items()):
        if set(clips)!=set(('neutral','angry')):continue
        if clips['neutral']['answer']!=clips['angry']['answer']:raise ValueError('Reference answers differ')
        pairs.append(dict(question_id=qid,voice=voice,clip_id=qid+'__'+voice,**clips))
    random.Random(42).shuffle(pairs)
    if count<0 or (count and count>len(pairs)):raise ValueError('Requested pair count unavailable')
    return pairs[:count] if count else pairs


def report(output):
    rows=read_rows(output/'results.jsonl')
    result=summarize(rows,CONDITIONS)
    print('Matched-state full activation replacement, not a learned vector. Requires angry audio for each question.')
    print('L31 answer patch is a final-layer positive control. Results are diagnostic, not deployment accuracy.')
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',type=Path,nargs='+',default=[Path('outputs/azure-jenny-next500'),Path('outputs/azure-jenny-remaining-test')])
    p.add_argument('--output',type=Path,default=Path('outputs/jenny-activation-patch100'))
    p.add_argument('--num-questions',type=int,default=100,help='0: all eligible pairs; default: 100 shuffled with seed42, without outcome filtering')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.summary_only:report(args.output);return
    pairs=select_pairs(args.paths,args.num_questions)
    if not pairs:raise ValueError('No matched recordings')
    plan=dict(pairs=pairs,prompt=PROMPT,seed=42,max_new_tokens=512,steps=128,block_length=32,
              temperature=0.,decoder='fast_dllm',targets=TARGETS,patch='full group replacement; donor uses current recipient answer embeddings')
    plan=json.loads(json.dumps(plan));config=args.output/'run_config.json'
    if config.exists() and json.loads(config.read_text())!=plan:raise ValueError('Patch inputs/settings changed; use another output')
    config.write_text(json.dumps(plan,indent=2)+'\n')
    print(f'{len(pairs)} fixed matched pairs; three patches plus neutral/angry references; seed42; no classifier/outcome filter.',flush=True)
    print('Each patched neutral forward performs one additional angry donor forward with identical answer state. One DIFFA model only.',flush=True)
    if args.prepare_only:return
    with (args.output/'queue.lock').open('w') as own:
        try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Activation patching is already queued or running')
        for kind in ('finalist-queue','azure-emotions'):
            subprocess.run([sys.executable,'-u','wait_for_experiments.py','--kind',kind],check=True)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for the shared GPU lock.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
            run(plan,args.output)


def run(plan,output):
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
    import src.fast_dllm.diffa_generate as decoder
    from extract_vectors import gen,load_model
    path=output/'results.jsonl';rows=read_rows(path,repair=True)
    if any(r.get('seed')!=42 or r.get('prompt')!=PROMPT for r in rows):raise ValueError('Cached recipe differs')
    complete={(r['clip_id'],r['condition']) for r in rows}
    pending=[p for p in plan['pairs'] if any((p['clip_id'],c['name']) not in complete for c in CONDITIONS)]
    if not pending:report(output);return
    model,tokenizer,features=load_model();llm=model.llm_model
    layers=llm.model.transformer.blocks;bounds={};original=decoder.insert_audio_embedding
    def insert(text,speech,lengths):
        start=int(lengths[0].item())
        bounds.update(start=start,acoustic_end=start+model.config.prompt_size,end=start+speech.shape[1])
        return original(text,speech,lengths)
    decoder.insert_audio_embedding=insert
    def generate(audio):
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        return gen(audio,16000,model,tokenizer,features,PROMPT,512)
    def save(pair,name,response,**extra):
        row=dict(clip_id=pair['clip_id'],question_id=pair['question_id'],condition=name,seed=42,prompt=PROMPT,
                 response=response,**score_response(response,pair['neutral']['answer']),**extra)
        with path.open('a') as stream:stream.write(json.dumps(row)+'\n')
        complete.add((pair['clip_id'],name));rows.append(row)
        print(f'  {name}: correct={row["correct"]} answer={row["predicted_answer"]}',flush=True)
    try:
        for index,pair in enumerate(pending,1):
            print(f'[{index}/{len(pending)}] {pair["question_id"]}',flush=True)
            audio={}
            for style in ('neutral','angry'):
                a,sr=sf.read(pair[style]['audio_file'],dtype='float32')
                if a.ndim==2:a=a.mean(axis=1)
                if sr!=16000:a=librosa.resample(a,orig_sr=sr,target_sr=16000)
                audio[style]=a
                name='baseline' if style=='neutral' else 'angry-unpatched'
                if (pair['clip_id'],name) not in complete:
                    started=time.monotonic();response=generate(a)
                    save(pair,name,response,generation_seconds=time.monotonic()-started)
            if all((pair['clip_id'],name) in complete for name,_,_ in TARGETS):continue
            donor,db=capture_prefix(llm,lambda:generate(audio['angry']),bounds)
            recipient,rb=capture_prefix(llm,lambda:generate(audio['neutral']),bounds)
            del recipient
            # Five fixed identity controls check that the replay machinery itself is neutral.
            audit_path=output/'identity-audit.jsonl'
            audited={r['clip_id'] for r in read_rows(audit_path)}
            if pair in plan['pairs'][:5] and pair['clip_id'] not in audited:
                self_prefix,sb=capture_prefix(llm,lambda:generate(audio['neutral']),bounds)
                with patch_forward(llm,layers[12],self_prefix,sb,rb,'acoustic',torch):
                    response=generate(audio['neutral'])
                baseline=next(r for r in rows if r['clip_id']==pair['clip_id'] and r['condition']=='baseline')
                if response!=baseline['response']:raise ValueError('Identity patch changed the neutral response; inspect before proceeding')
                with audit_path.open('a') as stream:stream.write(json.dumps(dict(clip_id=pair['clip_id'],passed=True))+'\n')
                del self_prefix
            for name,layer,group in TARGETS:
                if (pair['clip_id'],name) in complete:continue
                started=time.monotonic()
                with patch_forward(llm,layers[layer],donor,db,rb,group,torch) as state:
                    response=generate(audio['neutral'])
                if layer==31:
                    angry=next(r for r in rows if r['clip_id']==pair['clip_id'] and r['condition']=='angry-unpatched')
                    if response!=angry['response']:raise ValueError('L31 final-layer positive control differs from unpatched angry generation; inspect before interpreting patches')
                save(pair,name,response,generation_seconds=time.monotonic()-started,
                     donor_forwards=state['donor_forwards'],recipient_forwards=state['recipient_forwards'],patched_forwards=state['patched_forwards'])
            del donor
    finally:decoder.insert_audio_embedding=original
    report(output)


if __name__=='__main__':main()
