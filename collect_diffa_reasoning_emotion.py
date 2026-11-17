"""DIFFA-2: SAME100 neutral-reasoning / neutral-answer-only / angry-reasoning.

Observe the FIRST native fast-dllm forward, before any answer tokens are committed.
"""
import argparse
import csv
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import subprocess
import time

from extract_audio_vectors import GROUPS, save_tensor
from kimi_bbh_layer_diagnostic import digest
from phi4_answer_only import DIRECTIVE
from run_reasoning_transfer import freeze
from steer import read_rows

PROTOCOL = 'diffa-same100-reasoning-emotion-initial-mask-v1'
CONDITIONS = ('neutral-reasoning','neutral-answer-only','angry-reasoning')


def direct_prompt(prompt):
    phrases=("Let's think step by step.",
        'Give concise step-by-step reasoning with only the necessary calculations.',
        'Give concise step-by-step reasoning with only the necessary deductions.')
    matches=[p for p in phrases if p in prompt]
    if len(matches)!=1 or prompt.count(matches[0])!=1:
        raise ValueError('Need one known reasoning instruction, preserving the source final-answer format.')
    return prompt.replace(matches[0],DIRECTIVE)


def prepare(args):
    path=args.output/'run_config.json'
    if path.exists():
        plan=json.loads(path.read_text())
        if plan['protocol']!=PROTOCOL or plan['source_run']!=str(args.source_run.absolute()):
            raise ValueError('Collection settings changed; use another output folder.')
        print('Reusing the SAME frozen100 DIFFA question pairs in all three conditions.',flush=True)
        return plan
    source_path=args.source_run/'run_config.json'
    source=json.loads(source_path.read_text())
    if (source['seed']!=42 or source['max_new_tokens']!=512 or source['steps']!=128
            or source['block_length']!=32 or source['decoder']!='fast_dllm'):
        raise ValueError('Expected the established seed42 /512-token /128-step /32-block DIFFA evaluation recipe.')
    prompt=source['prompt'];answer_only=direct_prompt(prompt)
    groups={}
    for row in source['recordings']:
        if row['condition'] not in ('neutral','angry'):continue
        style=row['condition'];key=row['question_id'],row['voice']
        if row['style']!=style:raise ValueError('Delivery label mismatch.')
        if style in groups.setdefault(key,{}):raise ValueError('Duplicate question/voice/delivery.')
        if str(row['answer'])!=str(source['bank'][row['question_id']]['answer']):raise ValueError('Gold reference differs.')
        groups[key][style]=row
    ids=sorted(k for k,g in groups.items() if set(g)=={'neutral','angry'})
    if len(ids)<100:raise ValueError('Need at least100 matched neutral/angry recordings in the source config.')
    random.Random(42).shuffle(ids);ids=ids[:100]
    pairs=[]
    for q,v in ids:
        clips=groups[q,v]
        for row in clips.values():
            if digest(row['audio_file'])!=row['audio_sha256']:raise ValueError('Original DIFFA audio changed or is missing.')
        pairs.append(dict(question_id=q,voice=v,**clips))
    plan=dict(protocol=PROTOCOL,model='zhoujiaming777/DIFFA-2',source_run=str(args.source_run.absolute()),
        source_config_sha256=digest(source_path),pairs=pairs,bank={q:source['bank'][q] for q,v in ids},
        questions=100,conditions=list(CONDITIONS),forwards=300,layers=list(range(32)),groups=GROUPS,
        reasoning_prompt=prompt,answer_only_prompt=answer_only,seed=42,max_new_tokens=512,steps=128,block_length=32,
        temperature=0.,decoder='fast_dllm',audio_encoding='Existing gen_chunked route: native up to30s; unchanged full-audio encoding above30s.',
        capture='FIRST native fast-dllm LLaDA forward, all512 answer-mask positions, before commitment. '
            'Block-output means for acoustic, semantic, combined audio, and answer positions at all32 layers.',
        selection='Same100 seed42-shuffled question/voice pairs in EVERY condition; no correctness, format, emotion-classifier or duration filtering.',
        definitions={'reasoning_minus_answer_only':'neutral-reasoning minus neutral-answer-only',
                     'angry_minus_neutral':'angry-reasoning minus neutral-reasoning',
                     'neutral_minus_angry':'exact negative of angry_minus_neutral'},
        caveat='Instruction-conditioned initial states, not proven internal reasoning. Both contrasts share the same '
            'neutral-reasoning baseline. Include cross-half descriptive comparisons. DIFFA decoder is bidirectional; '
            'audio-token states can reflect the instruction. This is an INITIAL-step diagnostic, not the existing '
            'steering vectors averaged over decoding forwards. No similarity p-values or causal claims.')
    freeze(path,plan)
    print('DIFFA-2: SAME100 questions ×3 conditions =300 initial masked-answer forwards; no full answer generations.',flush=True)
    return plan


class InitialForwardCaptured(Exception):pass


def capture_initial(model,generate,decoder):
    """Enter the original decoder, observe one full forward, stop before token selection."""
    import torch
    llm=model.llm_model;layers=llm.model.transformer.blocks
    bounds={};states={};metadata={};handles=[];original=decoder.insert_audio_embedding
    def insert(text,speech,lengths):
        start=int(lengths[0].item())
        bounds.update(start=start,acoustic_end=start+model.config.prompt_size,end=start+speech.shape[1])
        return original(text,speech,lengths)
    def before(module,args,kwargs):
        if args or set(kwargs)!={'inputs_embeds'}:raise ValueError('Expected native fast-dllm inputs_embeds-only forward.')
        h=kwargs['inputs_embeds']
        if h.shape[0]!=1 or h.shape[1]<=512 or not bounds or bounds['end']>h.shape[1]-512:
            raise ValueError('Native speech bounds or512-answer span differs from the established decoder.')
        tail=h[:,-512:].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        metadata.update(input_tokens=h.shape[1],answer_tokens=512,answer_input_sha256=hashlib.sha256(tail).hexdigest(),
                        speech_bounds=dict(bounds),initial_answer_dtype=str(h.dtype))
    def hook(index):
        def observe(module,args,result):
            h=result[0]
            if index in states:raise ValueError('A block ran twice before initial-forward interception.')
            ranges=(slice(bounds['start'],bounds['acoustic_end']),slice(bounds['acoustic_end'],bounds['end']),
                    slice(bounds['start'],bounds['end']),slice(-512,None))
            if any(h[0,r].shape[0]==0 for r in ranges):raise ValueError('Empty DIFFA token group.')
            states[index]=torch.stack([h[0,r].detach().float().mean(0).cpu() for r in ranges])
        return observe
    def stop(module,args,result):
        if len(states)!=len(layers):raise ValueError('Not all decoder layers were captured.')
        raise InitialForwardCaptured()
    decoder.insert_audio_embedding=insert
    try:
        handles.append(llm.register_forward_pre_hook(before,with_kwargs=True))
        handles.extend(layer.register_forward_hook(hook(i)) for i,layer in enumerate(layers))
        handles.append(llm.register_forward_hook(stop))
        try:generate()
        except InitialForwardCaptured:pass
        else:raise ValueError('Native initial decoder forward was not intercepted.')
    finally:
        decoder.insert_audio_embedding=original
        for handle in handles:handle.remove()
    return dict(states=torch.stack([states[i] for i in range(len(layers))]),**metadata)


def collect(plan,output):
    import numpy as np
    import soundfile as sf
    import torch
    import src.fast_dllm.diffa_generate as decoder
    from extract_vectors import gen_chunked,load_model
    folder=output/'states';folder.mkdir(exist_ok=True);pending=[]
    for i,pair in enumerate(plan['pairs']):
        for condition in CONDITIONS:
            row=pair['angry' if condition=='angry-reasoning' else 'neutral']
            prompt=plan['answer_only_prompt'] if condition=='neutral-answer-only' else plan['reasoning_prompt']
            path=folder/f'{i:03d}-{condition}.pt'
            metadata=dict(protocol=PROTOCOL,question_id=pair['question_id'],voice=pair['voice'],condition=condition,
                          audio_sha256=row['audio_sha256'],prompt=prompt)
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=True)
                if any(saved[k]!=v for k,v in metadata.items()):raise ValueError('State cache differs from frozen collection.')
            else:pending.append((i,row,prompt,path,metadata))
    print(f'{len(pending)}/300 DIFFA initial forwards pending.',flush=True)
    if not pending:return
    model,tokenizer,features=load_model()
    if len(model.llm_model.model.transformer.blocks)!=32:raise ValueError('Expected32 DIFFA language decoder blocks.')
    (output/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,transformers=version('transformers'),
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),decoder_file=decoder.__file__,
        decoder_sha256=digest(decoder.__file__),generation_helper_sha256=digest(Path(__file__).with_name('extract_vectors.py')),
        model_config=model.config.to_dict(),generation='Only first fast-dllm forward; captured before committing tokens.'),indent=2,default=str)+'\n')
    for count,(i,row,prompt,path,metadata) in enumerate(pending,1):
        if digest(row['audio_file'])!=row['audio_sha256']:raise ValueError('Audio changed after selection.')
        audio,rate=sf.read(row['audio_file'],dtype='float32')
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        print(f'Collect [{count}/{len(pending)}] question{i+1}/100 {metadata["question_id"]} {metadata["condition"]}',flush=True)
        trace=capture_initial(model,lambda:gen_chunked(audio,rate,model,tokenizer,features,prompt,512,30),decoder)
        save_tensor(torch,dict(metadata,**trace),path)
        completed=[folder/f'{i:03d}-{c}.pt' for c in CONDITIONS]
        if all(p.exists() for p in completed):
            hashes={torch.load(p,map_location='cpu',weights_only=True)['answer_input_sha256'] for p in completed}
            if len(hashes)!=1:raise ValueError('Initial answer-mask embeddings differ across the three conditions; inspect before interpreting vectors.')


def summarize(output):
    import torch
    from collect_phi4_reasoning_emotion import cosine
    path=output/'run_config.json'
    if not path.exists():print('DIFFA collection not prepared.');return
    plan=json.loads(path.read_text());reason=[];emotion=[]
    for i,pair in enumerate(plan['pairs']):
        paths=[output/'states'/f'{i:03d}-{c}.pt' for c in CONDITIONS]
        if not all(p.exists() for p in paths):continue
        traces=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
        for t,c in zip(traces,CONDITIONS):
            if t['protocol']!=PROTOCOL or t['question_id']!=pair['question_id'] or t['condition']!=c:
                raise ValueError('Cache differs from frozen question/condition.')
        if len({t['answer_input_sha256'] for t in traces})!=1:raise ValueError('Initial answer states are not matched.')
        nr,na,ar=[t['states'] for t in traces]
        reason.append(nr-na);emotion.append(ar-nr)
    n=len(reason);print(f'DIFFA SAME100: {n}/100 matched triples complete.',flush=True)
    if not n:return
    r,e=torch.stack(reason),torch.stack(emotion);rm,em=r.mean(0),e.mean(0)
    save_tensor(torch,dict(protocol=PROTOCOL,pairs=n,groups=GROUPS,reasoning_minus_answer_only=rm,
        angry_minus_neutral=em,neutral_minus_angry=-em),output/('style-vectors.pt' if n==100 else 'style-vectors-partial.pt'))
    rows=[];print('Group      Layer  Angry-neutral vs reasoning  Neutral-angry vs reasoning',flush=True)
    for g,group in enumerate(GROUPS):
        for layer in range(rm.shape[0]):
            a,b=rm[layer,g],em[layer,g];cos=cosine(a,b);den=float(a.square().sum())
            row=dict(group=group,layer=layer,pairs=n,cosine_angry_neutral_vs_reasoning=cos,
                cosine_neutral_angry_vs_reasoning=None if cos is None else -cos,reasoning_norm=float(a.norm()),
                emotion_norm=float(b.norm()),signed_projection_coefficient=None if den<1e-12 else float(torch.dot(a,b)/den),
                squared_norm_overlap=None if cos is None else cos*cos,reasoning_split_half=None,emotion_split_half=None,
                cross_half_reason_even_emotion_odd=None,cross_half_reason_odd_emotion_even=None)
            if n>=4:
                r0,r1=r[::2,layer,g].mean(0),r[1::2,layer,g].mean(0)
                e0,e1=e[::2,layer,g].mean(0),e[1::2,layer,g].mean(0)
                row.update(reasoning_split_half=cosine(r0,r1),emotion_split_half=cosine(e0,e1),
                    cross_half_reason_even_emotion_odd=cosine(r0,e1),cross_half_reason_odd_emotion_even=cosine(r1,e0))
            rows.append(row)
            def fmt(v):return 'undefined' if v is None else f'{v:+.4f}'
            print(f'{group:10} {layer:5d} {fmt(cos):>30} {fmt(row["cosine_neutral_angry_vs_reasoning"]):>27}',flush=True)
    with (output/'layer-cosines.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(dict(protocol=PROTOCOL,pairs=n,layers=rows,note=plan['caveat'],
        reverse_direction='Exact negative, not a second independent experiment.'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','collect','summary'),default='collect')
    p.add_argument('--source-run',type=Path,default=Path('outputs/reasoning-transfer/gsm8k'))
    p.add_argument('--output',type=Path,default=Path('outputs/diffa-reasoning-emotion100'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='summary':summarize(args.output);return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=prepare(args)
        if args.mode=='prepare':return
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
            collect(plan,args.output)


if __name__=='__main__':main()
