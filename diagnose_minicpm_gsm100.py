"""Native MiniCPM GSM diagnostics: SAME100, three prompts, all-layer states and answers.

Concurrent by design: no shared GPU lock or queue waits; bounded Torch allocator.
"""
import argparse
from contextlib import contextmanager
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import shutil
import statistics

from audit_reasoning_emotion import geometry
from bbh_candidate_adapters import MODELS, checkpoint, load_audio
from collect_phi4_reasoning_emotion import CONDITIONS, PROTOCOL as PHI_PROTOCOL
from kimi_bbh_layer_diagnostic import atomic_save, digest, freeze

PROTOCOL = 'minicpm-gsm-same100-native-three-condition-v1'
SPEC = MODELS['minicpm']
SHAPE = (36, 2, 4096)


def prepare(args):
    config = args.output/'run_config.json'
    if config.exists():
        plan = json.loads(config.read_text())
        if (plan['protocol'] != PROTOCOL or plan['parent'] != str(args.parent.absolute())
                or plan['gpu_budget_gib'] != args.gpu_budget_gib or plan['recipe'] != SPEC): raise ValueError('Use a separate output for changed settings.')
        return plan
    source = args.parent/'run_config.json'; old = json.loads(source.read_text())
    if old['protocol'] != PHI_PROTOCOL or len(old['pairs']) != 100:
        raise ValueError('Need the original complete Phi SAME100 GSM question selection.')
    if len({p['question_id'] for p in old['pairs']}) != 100: raise ValueError('Expected100 unique questions.')
    plan = dict(protocol=PROTOCOL, parent=str(args.parent.absolute()), parent_sha256=digest(source),
                recipe=SPEC, pairs=old['pairs'], bank=old['bank'], seed=42, max_new_tokens=512,
                reasoning_prompt=old['reasoning_prompt'], answer_only_prompt=old['answer_only_prompt'],
                gpu_budget_gib=args.gpu_budget_gib, conditions=list(CONDITIONS), groups=['answer', 'audio'],
                collection='300 full native greedy text answers; first-forward states at all36 decoder blocks plus '
                    'means of answer-prediction states across native generation forwards. Native audio positions from '
                    'get_omni_embedding audio_bounds; prefix audio states observed on initial forward only.',
                selection='Exact same100 Phi question/voice pairs, no outcome/classifier/length filtering; no TTS or steering.',
                native_options=dict(enable_thinking=False, use_tts_template=True, generate_audio=False, do_sample=False,
                    max_new_tokens=512, min_new_tokens=0, num_beams=1, repetition_penalty=1., max_inp_length=32768, stream=False),
                note='Reasoning request is varied; native thinking mode stays disabled in ALL conditions. '
                    'Initial-state contrast is instruction-related, not proof of reasoning effort. Whole-generation '
                    'means use different generated histories and are reported separately. Audio/text template differs '
                    'from Phi; cross-model magnitudes are not matched measurements. Concurrent runtimes are not speed benchmarks.')
    plan['fingerprint']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    freeze(config, plan)
    print('Frozen same100 questions ×3 conditions:300 answers; native all36-layer diagnostics; no TTS/steering.', flush=True)
    return plan


def preflight_disk():
    import os
    root = Path(os.environ.get('HF_HUB_CACHE', '/workspace/.hf_home/hub')); root.mkdir(parents=True, exist_ok=True)
    snapshot = root/('models--'+SPEC['model'].replace('/', '--'))/'snapshots'/SPEC['revision']
    present = sum(p.stat().st_size for p in snapshot.rglob('*.safetensors') if p.is_file())
    env_root = Path('.venv-bbh-minicpm')
    # Setup can finish installing packages but fail before writing its final
    # requirements receipt. Already installed bytes are already excluded from
    # disk free space; reserve only the estimated remainder on a resumed setup.
    installed = sum(p.stat().st_size for p in env_root.rglob('*')
                    if not p.is_symlink() and p.is_file()) if env_root.exists() else 0
    environment = 0 if (env_root/'experiment-requirements.txt').exists() else max(0, 6*2**30-installed)
    needed = max(0, SPEC['weights_gib']*2**30-present)+environment+4.5*2**30
    free = shutil.disk_usage(root).free
    if free < needed: raise RuntimeError(f'Need approximately{needed/2**30:.1f} GiB free for missing MiniCPM weights/environment/results; have{free/2**30:.1f}. No data removed.')
    print(f'Disk preflight: {free/2**30:.1f} GiB free; estimated requirement{needed/2**30:.1f} GiB '
          f'(remaining environment{environment/2**30:.1f} GiB).', flush=True)


def cap_gpu(budget):
    import torch
    free, total = torch.cuda.mem_get_info(0)
    if budget <= 0 or budget*2**30 >= total: raise ValueError('Invalid GPU allocator budget.')
    if free < (budget+2)*2**30:
        raise RuntimeError(f'Only{free/2**30:.1f} GiB free; require{budget+2:.1f}. Exiting without waiting or stopping Phi.')
    torch.cuda.set_per_process_memory_fraction(budget*2**30/total, device=0)
    print(f'Concurrent start: {free/2**30:.1f} GiB free; Torch allocation cap{budget:.1f} GiB; no shared GPU lock.', flush=True)


@contextmanager
def observe(model):
    import torch
    llm = model.llm; layers = llm.model.layers; bounds = []; first = {}; sums = {}; counts = {}; meta = {}
    handles = []; original = model.get_omni_embedding; step = [0]; current = {}
    def embedding(data, *args, **kwargs):
        raw = data['audio_bounds']
        if len(raw) != 1: raise ValueError('Expected single native audio sequence.')
        bounds[:] = [(int(b[0]), int(b[1])) for b in raw[0]]
        return original(data, *args, **kwargs)
    def before(module, args, kwargs):
        if step[0] != 0: current['final'] = -1; return
        h = kwargs.get('inputs_embeds')
        if h is None or h.ndim != 3 or h.shape[0] != 1 or not bounds:
            raise ValueError('Expected native initial audio/text embedding forward.')
        length = h.shape[1]; mask = torch.zeros(length, dtype=torch.bool, device=h.device)
        for start, end in bounds:
            if not 0 <= start < end <= length: raise ValueError('Invalid native audio bounds.')
            if mask[start:end].any(): raise ValueError('Overlapping audio bounds.')
            mask[start:end] = True
        attention = kwargs.get('attention_mask')
        valid = torch.ones(length, dtype=torch.bool, device=h.device) if attention is None else attention[0].bool()
        if valid.shape != mask.shape or not valid.any() or (mask&~valid).any(): raise ValueError('Native attention/audio masks differ.')
        current.update(mask=mask, final=int(valid.nonzero()[-1].item()))
        if mask[current['final']]: raise ValueError('Answer-prediction position overlaps audio.')
        meta.update(input_tokens=length, audio_tokens=int(mask.sum()), audio_bounds=bounds.copy())
    def hook(index):
        def capture(module, inputs, result):
            h = result[0] if isinstance(result, (tuple, list)) else result
            value = h[0, current['final']].detach().float().cpu().clone()
            sums[index] = value if index not in sums else sums[index]+value
            counts[index] = counts.get(index, 0)+1
            if step[0] == 0:
                audio = h[0, current['mask']].detach().float().mean(0).cpu()
                first[index] = torch.stack((value, audio))
        return capture
    def after(module, inputs, result): step[0] += 1
    model.get_omni_embedding = embedding
    try:
        handles.append(llm.register_forward_pre_hook(before, with_kwargs=True))
        handles.extend(layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers))
        handles.append(llm.register_forward_hook(after))
        yield first, sums, counts, meta
    finally:
        model.get_omni_embedding = original
        for handle in handles: handle.remove()


def state_path(output, i, condition): return output/'states'/f'{i:03d}-{condition}.pt'


def metadata(plan, pair, condition):
    row = pair['angry' if condition == 'angry-reasoning' else 'neutral']
    return dict(protocol=PROTOCOL, question_id=pair['question_id'], voice=pair['voice'], condition=condition,
                run_fingerprint=plan['fingerprint'],model=plan['recipe']['model'],revision=plan['recipe']['revision'],
                seed=42,max_new_tokens=512,
                audio_sha256=row['audio_sha256'], prompt=plan['answer_only_prompt'] if condition=='neutral-answer-only' else plan['reasoning_prompt'])


def validate(saved, expected):
    import torch
    if any(saved.get(k) != v for k, v in expected.items()): raise ValueError('Cache metadata differs from frozen experiment.')
    if (tuple(saved['states'].shape) != SHAPE or tuple(saved['generation_answer_mean'].shape) != (SHAPE[0], SHAPE[2])
            or not torch.isfinite(saved['states']).all() or not torch.isfinite(saved['generation_answer_mean']).all()
            or saved['forwards'] <= 0): raise ValueError('Invalid activation cache.')


def collect(plan, output):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor
    torch.set_num_threads(min(4, torch.get_num_threads()))
    pending=[]
    for i,pair in enumerate(plan['pairs']):
        for c in CONDITIONS:
            path=state_path(output,i,c); expected=metadata(plan,pair,c)
            if path.exists(): validate(torch.load(path,map_location='cpu',weights_only=True),expected)
            else: pending.append((i,pair,c,path,expected))
    print(f'{len(pending)}/300 native generations pending.',flush=True)
    if not pending:return
    if version('transformers') != SPEC['transformers']:raise ValueError('Use .venv-bbh-minicpm.')
    path=checkpoint('minicpm')
    cap_gpu(plan['gpu_budget_gib'])
    model=AutoModel.from_pretrained(path,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',
                                    init_vision=False,init_audio=True,init_tts=False).eval().cuda()
    processor=AutoProcessor.from_pretrained(path,trust_remote_code=True)
    if len(model.llm.model.layers)!=SHAPE[0] or model.llm.config.hidden_size!=SHAPE[2]:raise ValueError('Unexpected pinned MiniCPM decoder layout.')
    (output/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,transformers=version('transformers'),
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),collector_sha256=digest(__file__),
        model_config=model.config.to_dict(),gpu_budget_gib=plan['gpu_budget_gib']),indent=2,default=str)+'\n')
    (output/'states').mkdir(exist_ok=True)
    for count,(i,pair,c,path,expected) in enumerate(pending,1):
        row=pair['angry' if c=='angry-reasoning' else 'neutral']
        if digest(row['audio_file'])!=row['audio_sha256']:raise ValueError('Recording changed.')
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        print(f'[{count}/{len(pending)}] {pair["question_id"]} {c}',flush=True)
        with torch.inference_mode(),observe(model) as (first,sums,counts,meta):
            response=model.chat(msgs=[{'role':'user','content':[expected['prompt'],load_audio(row['audio_file'])]}],
                                processor=processor,**plan['native_options'])
        if not isinstance(response,str) or len(first)!=SHAPE[0] or len(set(counts.values()))!=1:
            raise ValueError('Incomplete native output/capture.')
        saved=dict(expected,states=torch.stack([first[l] for l in range(SHAPE[0])]),
            generation_answer_mean=torch.stack([sums[l]/counts[l] for l in range(SHAPE[0])]),forwards=counts[0],
            response=response,response_tokens=len(processor.tokenizer.encode(response,add_special_tokens=False)),**meta)
        validate(saved,expected);atomic_save(path,saved)
        print(f'  {saved["response_tokens"]} text tokens; {saved["forwards"]} native forwards.',flush=True)


def summarize(plan,output):
    import torch
    torch.set_num_threads(min(4,torch.get_num_threads()))
    triples=[];answers=[]
    for i,pair in enumerate(plan['pairs']):
        paths=[state_path(output,i,c) for c in CONDITIONS]
        if not all(p.exists() for p in paths):continue
        traces=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
        for t,c in zip(traces,CONDITIONS):
            validate(t,metadata(plan,pair,c));answers.append({k:v for k,v in t.items() if k not in ('states','generation_answer_mean')})
        triples.append(traces)
    print(f'MiniCPM diagnostics: {len(triples)}/100 complete triples.',flush=True)
    if not triples:return
    n,na,a=[torch.stack([t[j]['states'].double() for t in triples]) for j in (0,1,2)]
    r=n-na;e=a-n;rm=r.mean(0);em=e.mean(0)
    gen=[torch.stack([t[j]['generation_answer_mean'].double() for t in triples]) for j in (0,1,2)]
    gr,ge=(gen[0]-gen[1]).mean(0),(gen[2]-gen[0]).mean(0)
    vector=dict(protocol=PROTOCOL,run_fingerprint=plan['fingerprint'],model=SPEC['model'],revision=SPEC['revision'],groups=['answer','audio'],
                question_ids=[t[0]['question_id'] for t in triples],reasoning_minus_answer_only=rm.float(),
                angry_minus_neutral=em.float(),generation_reasoning_minus_answer_only=gr.float(),
                generation_angry_minus_neutral=ge.float())
    atomic_save(output/('style-vectors.pt' if len(triples)==100 else 'style-vectors-partial.pt'),vector)
    rows=[]
    print('Answer positions: layer / anger vs reasoning / cross-half A-B / cross-half B-A / relative drift %')
    for g,group in enumerate(('answer','audio')):
        for l in range(SHAPE[0]):
            geo=geometry(rm[l,g],em[l,g]);c0=geometry(r[::2,l,g].mean(0),e[1::2,l,g].mean(0))['cosine'] if len(triples)>=4 else None
            c1=geometry(r[1::2,l,g].mean(0),e[::2,l,g].mean(0))['cosine'] if len(triples)>=4 else None
            norms=n[:,l,g].norm(dim=-1);valid=norms>0
            if not valid.any(): raise ValueError('Undefined relative drift: all neutral representations have zero norm.')
            drift=100*e[:,l,g].norm(dim=-1)[valid]/norms[valid]
            anorm=a[:,l,g].norm(dim=-1);both=valid&(anorm>0)
            representation_cosine=(n[:,l,g]*a[:,l,g]).sum(-1)[both]/(norms[both]*anorm[both])
            dn=e[:,l,g].norm(dim=-1);units=e[:,l,g]/dn.clamp_min(1e-30).unsqueeze(-1)
            row=dict(group=group,layer=l,pairs=len(triples),cosine=geo['cosine'],signed_projection_coefficient=geo['coefficient'],
                squared_norm_overlap=geo['overlap'],cross_half_A_B=c0,cross_half_B_A=c1,
                mean_relative_drift_pct=float(drift.mean()),median_relative_drift_pct=float(drift.quantile(.5)),
                mean_representation_cosine=float(representation_cosine.mean()),shift_consistency=float(units.mean(0).norm()),
                reason_split_half=geometry(r[::2,l,g].mean(0),r[1::2,l,g].mean(0))['cosine'] if len(triples)>=4 else None,
                emotion_split_half=geometry(e[::2,l,g].mean(0),e[1::2,l,g].mean(0))['cosine'] if len(triples)>=4 else None,
                generation_answer_cosine=geometry(gr[l],ge[l])['cosine'] if g==0 else None)
            rows.append(row)
            def fmt(v):return 'undefined' if v is None else f'{v:+.4f}'
            if g==0:print(f'{l:5d} {fmt(geo["cosine"]):>12} {fmt(c0):>14} {fmt(c1):>14} {float(drift.mean()):12.2f}',flush=True)
    for group in ('answer','audio'):
        print(f'\nTop5 initial activation drift: {group} positions')
        print('Layer  Mean change %  Median change %  Mean cosine  Shift consistency')
        for row in sorted((t for t in rows if t['group']==group),key=lambda t:t['mean_relative_drift_pct'],reverse=True)[:5]:
            print(f'{row["layer"]:5d} {row["mean_relative_drift_pct"]:14.2f} {row["median_relative_drift_pct"]:16.2f}'
                  f' {row["mean_representation_cosine"]:12.4f} {row["shift_consistency"]:18.4f}')
    with (output/'layer-diagnostics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'results.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in answers))
    (output/'summary.json').write_text(json.dumps(dict(protocol=PROTOCOL,pairs=len(triples),layers=rows,note=plan['note']),indent=2)+'\n')


def score(plan,output):
    from rescore_audio_steering import HarnessTaskScorer,numeric_equal,paired
    if not (output/'results.jsonl').exists():print('Run --mode summary first to gather complete triples.');return
    rows=[json.loads(l) for l in (output/'results.jsonl').read_text().splitlines() if l.strip()]
    if not rows:return
    scorer=HarnessTaskScorer('gsm8k',output/'harness/upstream')
    docs=[dict(question=plan['bank'][r['question_id']]['question'],answer=plan['bank'][r['question_id']]['answer']) for r in rows]
    values=scorer.score([r['response'] for r in rows],docs)
    for i,r in enumerate(rows):r['scores']={**{m:v[i]['correct'] for m,v in values.items()},
        'flexible-numeric':numeric_equal(values['flexible-extract'][i]['prediction'],docs[i]['answer']),
        'format-compliant':values['strict-match'][i]['prediction']!='[invalid]'}
    groups={c:{r['question_id']:r for r in rows if r['condition']==c} for c in CONDITIONS}
    keys=set.intersection(*(set(g) for g in groups.values()));comparisons=[];accuracy=[]
    if not keys:print('No matched triples scored yet.');return
    print(f'\nGSM8K: {len(keys)} identical questions in all three conditions.')
    for c in CONDITIONS:
        t=[groups[c][k] for k in keys];lengths=[r['response_tokens'] for r in t]
        accuracy.append(dict(condition=c,pairs=len(t),correct={m:sum(r['scores'][m] for r in t)
            for m in ('flexible-numeric','strict-match','format-compliant')},mean_tokens=statistics.mean(lengths),median_tokens=statistics.median(lengths)))
        print(c, 'numeric/strict/format:',*[f'{sum(r["scores"][m] for r in t)/len(t):.1%}' for m in ('flexible-numeric','strict-match','format-compliant')],
              'mean/median tokens:',round(statistics.mean(lengths),1),statistics.median(lengths))
    for c in ('angry-reasoning','neutral-answer-only'):
        for m in ('flexible-numeric','strict-match','format-compliant'):
            result=paired([(groups['neutral-reasoning'][k]['scores'][m],groups[c][k]['scores'][m]) for k in sorted(keys)])
            comparisons.append(dict(condition=c,metric=m,**result));print(c,m,f'gain={result["gain_pp"]:+.2f} pp; p={result["mcnemar_exact_p"]:.4g}; CI={result["ci95_gain_pp"]}')
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (output/'accuracy-summary.json').write_text(json.dumps(dict(accuracy=accuracy,comparisons=comparisons,scorer=scorer.provenance,
        note='Audio-adapted native inference, upstream GSM8K saved-answer scoring; numeric equivalence additional; exploratory/unadjusted.'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('prepare','collect','summary','score','disk'),default='collect')
    p.add_argument('--parent',type=Path,default=Path('outputs/phi4-reasoning-emotion100-gsm8k'))
    p.add_argument('--output',type=Path,default=Path('outputs/minicpm-gsm100-diagnostics'))
    p.add_argument('--gpu-budget-gib',type=float,default=22.)
    args=p.parse_args()
    if args.mode=='disk':preflight_disk();return
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode in ('summary','score'):
        plan=json.loads((args.output/'run_config.json').read_text())
        if args.mode=='summary':summarize(plan,args.output)
        else:score(plan,args.output)
        return
    import fcntl
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=prepare(args)
        if args.mode=='prepare':return
        collect(plan,args.output)


if __name__=='__main__':main()
