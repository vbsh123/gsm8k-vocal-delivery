"""Same 100 questions, three Phi-4 prefills: neutral reasoning/answer-only, angry reasoning.

Collect prompt-conditioned states BEFORE any answer generation. No interventions.
"""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

from bbh_candidate_adapters import MODELS, checkpoint, load_audio, phi_prompt
from kimi_bbh_layer_diagnostic import atomic_save, digest, freeze
from pack_experiment_backup import snapshot
from phi4_answer_only import answer_only_prompt
from steer import read_rows

PROTOCOL = 'phi4-same100-reasoning-emotion-prefill-v1'
CONDITIONS = ('neutral-reasoning', 'neutral-answer-only', 'angry-reasoning')


def source_run(args):
    if args.source_run is not None: return args.source_run
    first = Path('outputs/phi4-answer-only-followup')/args.task/'original'
    fallback = Path('outputs/gsm8k-candidates/phi4/original' if args.task == 'gsm8k' else
                    'outputs/bbh-three-candidates/phi4/original')
    return first if (first/'run_config.json').exists() else fallback


def prepare(args):
    source = source_run(args); config = args.output/'run_config.json'
    if config.exists():
        plan = json.loads(config.read_text())
        if plan['protocol'] != PROTOCOL or plan['source_run'] != str(source.absolute()) or plan['task'] != args.task:
            raise ValueError('Collection settings changed; use a separate output folder.')
        print('Reusing the same frozen 100 questions for ALL three conditions.', flush=True)
        return plan
    parent_path = source/'run_config.json'; parent = json.loads(parent_path.read_text())
    if parent['recipe']['model'] != MODELS['phi4']['model'] or parent['recipe']['revision'] != MODELS['phi4']['revision']:
        raise ValueError('Expected pinned Phi-4 source run.')
    reasoning_prompt = parent['prompt']; direct_prompt = answer_only_prompt(args.task, reasoning_prompt)
    data, trimmed = snapshot(source/'results.jsonl')
    rows = [json.loads(s) for s in data.splitlines() if s.strip()]
    records = {(r['question_id'], r['voice'], r['style']): r for r in parent['recordings']}
    completed = set()
    for row in rows:
        key = row['question_id'], row['voice'], row['style']
        if (key not in records or row['prompt'] != reasoning_prompt or row['seed'] != 42 or row['max_new_tokens'] != 512
                or row['model'] != MODELS['phi4']['model'] or row['revision'] != MODELS['phi4']['revision']
                or row['audio_sha256'] != records[key]['audio_sha256'] or row['interventions']):
            raise ValueError('Source answer differs from frozen model/prompt/audio settings.')
        completed.add(key)
    ids = sorted({(q,v) for q,v,s in completed if all((q,v,t) in completed for t in ('neutral','angry'))})
    if len(ids) < 100: raise ValueError('Need at least 100 completed neutral/angry question pairs.')
    random.Random(42).shuffle(ids); ids = ids[:100]
    pairs = [dict(question_id=q, voice=v, neutral=records[q,v,'neutral'], angry=records[q,v,'angry']) for q,v in ids]
    plan = dict(protocol=PROTOCOL, task=args.task, source_run=str(source.absolute()), source_config_sha256=digest(parent_path),
        source_answers_snapshot_sha256=hashlib.sha256(data).hexdigest(), incomplete_final_line_omitted=trimmed,
        model=MODELS['phi4']['model'], revision=MODELS['phi4']['revision'], transformers=MODELS['phi4']['transformers'],
        seed=42, questions=100, forwards=300, layers=list(range(32)), groups=['answer', 'audio'], pairs=pairs,
        bank={q: parent['bank'][q] for q,v in ids}, reasoning_prompt=reasoning_prompt, answer_only_prompt=direct_prompt,
        selection='Same 100 seed42-shuffled completed pairs in every condition; no correctness, length, format or classifier filter.',
        conditions=list(CONDITIONS), collection='Native Phi-4 forward at original audio/text prompt; last valid prompt position '
            'at each decoder block predicts the first answer token. Audio-token means also retained. No generated text, '
            'answer histories, KV reuse across conditions, or steering.',
        definitions={'reasoning_minus_answer_only':'neutral-reasoning minus neutral-answer-only',
                     'angry_minus_neutral':'angry-reasoning minus neutral-reasoning',
                     'neutral_minus_angry':'exact negative of angry_minus_neutral'},
        caveat='Reasoning-INSTRUCTION direction, not established internal reasoning. Emotion and instruction contrasts share '
               'the same neutral-reasoning baseline, which can influence cosine. Report cross-half diagnostics too. '
               'Audio tokens precede the instruction and cannot causally attend to it; their instruction contrast should be near zero.')
    freeze(config, plan)
    print('Frozen SAME 100 questions × 3 conditions = 300 prefills; no answer generation.', flush=True)
    return plan


def decoder_layers(model):
    layers = model.model.layers
    if len(layers) != 32 or model.config.hidden_size != 3072:
        raise ValueError('Phi-4 language decoder differs from pinned 32-layer / 3072-wide architecture.')
    return layers


def capture(model, inputs):
    import torch
    layers = decoder_layers(model)
    ids = inputs['input_ids'][0]
    audio_mask = (ids == 200011) | (ids <= -10000)
    if not audio_mask.any(): raise ValueError('No native Phi-4 audio-token positions found.')
    if int(inputs['audio_embed_sizes'].sum()) != int(audio_mask.sum()):
        raise ValueError('Native audio token count differs from encoded audio sizes.')
    valid = inputs['attention_mask'][0].bool().nonzero().flatten()
    final = int(valid[-1]); states = {}; handles = []
    def hook(index):
        def collect(module, args, result):
            h = result[0] if isinstance(result, (tuple,list)) else result
            if index in states: raise ValueError('A decoder block ran twice in one prefill.')
            states[index] = dict(answer=h[0,final].detach().float().cpu().clone(),
                                 audio=h[0,audio_mask].detach().float().mean(0).cpu().clone())
        return collect
    try:
        for i, layer in enumerate(layers): handles.append(layer.register_forward_hook(hook(i)))
        with torch.inference_mode():
            output = model(**inputs, use_cache=False, return_dict=True, num_logits_to_keep=1)
        next_id = int(output.logits[0,-1].argmax())
    finally:
        for handle in handles: handle.remove()
    if len(states) != len(layers): raise ValueError('Missing decoder-layer captures.')
    return dict(answer=torch.stack([states[i]['answer'] for i in range(len(layers))]),
                audio=torch.stack([states[i]['audio'] for i in range(len(layers))]),
                input_tokens=len(ids), audio_tokens=int(audio_mask.sum()), first_answer_argmax_id=next_id,
                input_ids_sha256=hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest())


def collect(plan, output):
    import numpy as np
    import torch
    from importlib.metadata import version
    from transformers import AutoModelForCausalLM, AutoProcessor
    if version('transformers') != plan['transformers']: raise ValueError('Use the existing .venv-bbh-phi4 environment.')
    folder = output/'states'; folder.mkdir(exist_ok=True)
    pending=[]
    for i,pair in enumerate(plan['pairs']):
        for condition in CONDITIONS:
            row = pair['angry' if condition == 'angry-reasoning' else 'neutral']
            prompt = plan['answer_only_prompt'] if condition == 'neutral-answer-only' else plan['reasoning_prompt']
            path = folder/f'{i:03d}-{condition}.pt'
            metadata = dict(protocol=PROTOCOL, question_id=pair['question_id'], voice=pair['voice'], condition=condition,
                            audio_sha256=row['audio_sha256'], prompt=prompt)
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=True)
                if any(saved[k] != v for k,v in metadata.items()): raise ValueError('Saved state cache differs from frozen question/condition.')
            else: pending.append((i,row,prompt,path,metadata))
    print(f'{len(pending)}/300 collection forwards pending.', flush=True)
    if not pending: return
    path=checkpoint('phi4')
    processor=AutoProcessor.from_pretrained(path,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True,torch_dtype=torch.bfloat16,
        device_map='cuda:0',_attn_implementation='sdpa').eval()
    (output/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,transformers=version('transformers'),
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),attention='sdpa',dtype='bfloat16',
        saved_states_dtype='float32',generation='none; native prefill only, use_cache=False'),indent=2)+'\n')
    for count,(i,row,prompt,path,metadata) in enumerate(pending,1):
        if digest(row['audio_file']) != row['audio_sha256']: raise ValueError('Audio changed since selection.')
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        inputs=processor(text=phi_prompt(prompt),audios=[(load_audio(row['audio_file']),16000)],return_tensors='pt').to(model.device,dtype=torch.bfloat16)
        print(f'Collect [{count}/{len(pending)}] question {i+1}/100 {metadata["question_id"]} {metadata["condition"]}',flush=True)
        result=capture(model,inputs)
        atomic_save(path,dict(metadata,**result))


def cosine(a,b):
    import torch
    norm=a.norm()*b.norm()
    return float(torch.dot(a,b)/norm) if float(norm)>1e-12 else None


def summarize(output):
    import torch
    path=output/'run_config.json'
    if not path.exists(): print('Collection not prepared.'); return
    plan=json.loads(path.read_text()); reason=[]; emotion=[]; audio_reason=[]; audio_emotion=[]
    for i,pair in enumerate(plan['pairs']):
        paths=[output/'states'/f'{i:03d}-{c}.pt' for c in CONDITIONS]
        if not all(p.exists() for p in paths): continue
        nr,na,ar=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
        for row,c in zip((nr,na,ar),CONDITIONS):
            if row['question_id'] != pair['question_id'] or row['condition'] != c or row['protocol'] != PROTOCOL:
                raise ValueError('State file differs from frozen collection.')
        reason.append(nr['answer']-na['answer']); emotion.append(ar['answer']-nr['answer'])
        audio_reason.append(nr['audio']-na['audio']); audio_emotion.append(ar['audio']-nr['audio'])
    n=len(reason); print(f'Same questions completed in ALL THREE conditions: {n}/100.',flush=True)
    if not n: return
    r,e,ra,ea=[torch.stack(v) for v in (reason,emotion,audio_reason,audio_emotion)]
    rm,em,ram,eam=[v.mean(0) for v in (r,e,ra,ea)]
    vectors=dict(protocol=PROTOCOL,pairs=n,layers=plan['layers'],
        reasoning_minus_answer_only=rm,angry_minus_neutral=em,neutral_minus_angry=-em,
        audio_reasoning_minus_answer_only=ram,audio_angry_minus_neutral=eam,audio_neutral_minus_angry=-eam)
    atomic_save(output/('style-vectors.pt' if n==100 else 'style-vectors-partial.pt'),vectors)
    rows=[]
    print('Layer  Angry-neutral vs reasoning  Neutral-angry vs reasoning  Reason split-half  Emotion split-half')
    for layer in range(rm.shape[0]):
        cos=cosine(em[layer],rm[layer]); den=float(rm[layer].square().sum())
        row=dict(layer=layer,pairs=n,cosine_angry_neutral_vs_reasoning=cos,
            cosine_neutral_angry_vs_reasoning=None if cos is None else -cos,
            reasoning_norm=float(rm[layer].norm()),emotion_norm=float(em[layer].norm()),
            signed_projection_coefficient=None if den<1e-12 else float(torch.dot(em[layer],rm[layer])/den),
            squared_norm_overlap=None if cos is None else cos*cos,
            reasoning_split_half=None,emotion_split_half=None,cross_half_reason_even_emotion_odd=None,
            cross_half_reason_odd_emotion_even=None,audio_instruction_difference_norm=float(ram[layer].norm()))
        if n>=4:
            r0,r1=r[::2,layer].mean(0),r[1::2,layer].mean(0)
            e0,e1=e[::2,layer].mean(0),e[1::2,layer].mean(0)
            row.update(reasoning_split_half=cosine(r0,r1),emotion_split_half=cosine(e0,e1),
                cross_half_reason_even_emotion_odd=cosine(r0,e1),cross_half_reason_odd_emotion_even=cosine(r1,e0))
        rows.append(row)
        def fmt(v): return 'undefined' if v is None else f'{v:+.4f}'
        print(f'{layer:5d} {fmt(cos):>31} {fmt(row["cosine_neutral_angry_vs_reasoning"]):>27} '
              f'{fmt(row["reasoning_split_half"]):>19} {fmt(row["emotion_split_half"]):>19}',flush=True)
    with (output/'layer-cosines.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(dict(protocol=PROTOCOL,pairs=n,layers=rows,
        note=plan['caveat'],reverse_direction='Exactly negative, not an independent experiment. '
        'Primary cosine uses answer-prediction states before generation; cross-half comparisons remove reuse '
        'of the same question baseline across contrasts. No similarity p-values or causal claims.'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','collect','summary'),default='collect')
    p.add_argument('--task',choices=('gsm8k','bbh3'),default='gsm8k')
    p.add_argument('--source-run',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.output is None: args.output=Path('outputs')/f'phi4-reasoning-emotion100-{args.task}'
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='summary': summarize(args.output);return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare(args)
        if args.mode=='prepare':return
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
            collect(plan,args.output)


if __name__=='__main__':main()
