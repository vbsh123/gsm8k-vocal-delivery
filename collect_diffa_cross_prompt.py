"""Complete the first-forward GSM8K angry × prompt comparison on the frozen 94 native pairs."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from collect_diffa_reasoning_emotion import capture_initial
from diffa_model_receipts import model_revisions
from phi4_queued_workers import gpu_queue
from phi4_sweep import job_lock
from winning_diffa_sweeps import atomic_text, digest, freeze, seed

REF = Path(__file__).resolve().parent/'data/diffa-cross-prompt94.npz'
OUT = Path('outputs/diffa-cross-prompt94')
CONDITIONS = ('neutral-reasoning', 'angry-reasoning', 'neutral-answer-only')


def build_reference(parent, dest):
    import torch
    plan = json.loads((parent/'run_config.json').read_text())
    rows, states = [], []
    for i, pair in enumerate(plan['pairs']):
        if any(pair[s]['seconds'] > 30 for s in ('neutral', 'angry')):
            continue
        traces = []
        for condition in CONDITIONS:
            src = parent/'states'/f'{i:03d}-{condition}.pt'
            t = torch.load(src, map_location='cpu', weights_only=True)
            style, prompt_type = condition.split('-', 1)
            prompt = plan['answer_only_prompt' if prompt_type=='answer-only' else 'reasoning_prompt']
            if (t['question_id'], t['voice'], t['audio_sha256'], t['condition'], t['prompt']) != (
                pair['question_id'], pair['voice'], pair[style]['audio_sha256'], condition, prompt):
                raise ValueError(f'Mismatched parent state: {src}')
            traces.append(t['states'][:,plan['groups'].index('acoustic')].float().numpy().copy())
        rows.append(dict(question_id=pair['question_id'], **{k:pair['angry'][k] for k in
                         ('audio_sha256','audio_file','seconds')}, neutral_sha256=pair['neutral']['audio_sha256']))
        states.append(traces)
    if len(rows)!=94 or len({r['question_id'] for r in rows})!=94:
        raise ValueError('Expected the 94 unique native GSM8K pairs.')
    split = torch.randperm(94, generator=torch.Generator().manual_seed(42)).numpy().reshape(2,47)
    full = np.stack(states)
    means = np.stack([full[ix].mean(0, dtype=np.float64) for ix in split])
    runtime = json.loads((parent/'runtime.json').read_text())
    meta = dict(rows=rows, conditions=list(CONDITIONS), parent_config_sha256=digest(parent/'run_config.json'),
                reasoning_prompt=plan['reasoning_prompt'], answer_only_prompt=plan['answer_only_prompt'],
                parent_decoder_sha256=runtime['decoder_sha256'], model=plan['model'])
    np.savez_compressed(dest, meta=json.dumps(meta), split=split, means=means.astype('float32'),
                        replay=full[0,1])


def audio_paths(rows, audio_root):
    from jenny_bundle_registry import sha
    paths, missing = {}, []
    for row in rows:
        source = Path(row['audio_file'])
        rel = source.relative_to('/workspace/gsm8k-vocal-delivery')
        candidates = [source, rel, Path('outputs/jenny-zip-audio')/rel]
        if audio_root:
            candidates = [audio_root/source.name, audio_root/rel, *candidates]
        path = next((p for p in candidates if p.is_file() and sha(p)==row['audio_sha256']), None)
        if path:
            paths[row['question_id']] = path
        else:
            missing.append(source.name)
    if missing:
        raise FileNotFoundError(f'{len(missing)} original angry WAV(s) missing or changed, e.g. {missing[0]}. '
                                'Use the machine with outputs/azure-kimi-gsm8k1269 or pass --audio-root DIR.')
    return paths


def capture(path, prompt, model, tokenizer, features, decoder):
    import soundfile as sf
    from extract_vectors import gen
    audio, rate = sf.read(path, dtype='float32')
    if len(audio)/rate > 30:
        raise ValueError(f'Expected a native-length recording: {path}')
    seed()
    trace = capture_initial(model, lambda:gen(audio,rate,model,tokenizer,features,prompt,512), decoder)
    return trace['states'][:,0].float().contiguous().numpy()


def write_state(path, x):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as out:
        np.save(out, x)
    tmp.replace(path)


def cos(a,b):
    return (a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)).clip(1e-30)


def compare(ref, new):
    means = ref['means'].astype('float64')
    r = means[:,1]-means[:,0]
    a = np.stack([new[ix].mean(0,dtype=np.float64) for ix in ref['split']])-means[:,2]
    result = dict(reasoning_split_cos=cos(r[0],r[1]), answer_only_split_cos=cos(a[0],a[1]),
                  reasoning_A_vs_answer_only_B=cos(r[0],a[1]), reasoning_B_vs_answer_only_A=cos(r[1],a[0]),
                  pooled_prompt_cos=cos(r.mean(0),a.mean(0)),
                  answer_only_over_reasoning_norm=np.linalg.norm(a.mean(0),axis=-1)/np.linalg.norm(r.mean(0),axis=-1))
    return [dict(layer=l, **{name:float(x[l]) for name,x in result.items()}) for l in range(r.shape[1])]


def report(folder, ref, rows):
    paths = [folder/'states'/(r['question_id']+'.npy') for r in rows]
    n = sum(p.is_file() for p in paths)
    print(f'Angry/answer-only: {n}/{len(rows)} saved initial forwards.',flush=True)
    if n < len(rows):
        return
    new = np.stack([np.load(p,allow_pickle=False) for p in paths])
    if new.shape != (94,*ref['replay'].shape) or not np.isfinite(new).all():
        raise ValueError('Damaged acoustic states.')
    comparisons = compare(ref,new)
    buf = io.StringIO()
    writer = csv.DictWriter(buf,fieldnames=comparisons[0])
    writer.writeheader()
    writer.writerows(comparisons)
    atomic_text(folder/'layer-comparison.csv',buf.getvalue())
    for l in (0,15,31):
        s = comparisons[l]
        print(f'L{l}: cross-prompt held-out cosine {s["reasoning_A_vs_answer_only_B"]:.4f} / '
              f'{s["reasoning_B_vs_answer_only_A"]:.4f}; within-prompt cosine '
              f'{s["reasoning_split_cos"]:.4f} / {s["answer_only_split_cos"]:.4f}; '
              f'answer-only / reasoning norm {s["answer_only_over_reasoning_norm"]:.3f}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['run','summary','prepare','build-reference'],default='run',nargs='?')
    p.add_argument('--parent',type=Path,default=Path('outputs/diffa-reasoning-emotion100'))
    p.add_argument('--reference',type=Path,default=REF)
    p.add_argument('--audio-root',type=Path)
    p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args()
    if args.mode=='build-reference':
        build_reference(args.parent,args.reference)
        return
    ref=np.load(args.reference,allow_pickle=False)
    meta=json.loads(str(ref['meta']))
    rows=meta['rows']
    folder=args.output
    config=dict(reference_sha256=digest(args.reference),condition='angry-answer-only',
                written_prompt=meta['answer_only_prompt'],question_ids=[r['question_id'] for r in rows])
    if args.mode=='summary':
        if (folder/'run_config.json').exists() and json.loads((folder/'run_config.json').read_text())!=config:
            raise ValueError('Reference differs from the frozen run.')
        report(folder,ref,rows)
        return
    with job_lock(folder):
        freeze(folder/'run_config.json',config)
        states=folder/'states'
        states.mkdir(exist_ok=True)
        pending=[r for r in rows if not (states/(r['question_id']+'.npy')).exists()]
        if pending:
            needed = pending if pending[0] is rows[0] else [rows[0], *pending]
            paths=audio_paths(needed,args.audio_root)
            print(f'{len(pending)} pending angry/answer-only forwards. Exact source audio verified.',flush=True)
        if args.mode=='prepare':
            return
        if pending:
            with gpu_queue():
                from extract_vectors import load_model
                import src.fast_dllm.diffa_generate as decoder
                model,tok,features=load_model()
                observed=capture(paths[rows[0]['question_id']],meta['reasoning_prompt'],model,tok,features,decoder)
                base=ref['replay']
                err=np.linalg.norm(observed-base,axis=1)/np.linalg.norm(base,axis=1)
                if not np.isfinite(err).all() or err.max()>0.02:
                    raise ValueError(f'Reasoning replay differs from archived activations (max relative error {err.max():.4f}).')
                atomic_text(folder/'replay.json',json.dumps(dict(question_id=rows[0]['question_id'],
                    relative_error_per_layer=err.tolist(),decoder_sha256=digest(decoder.__file__),
                    parent_decoder_sha256=meta['parent_decoder_sha256'],model_revisions=model_revisions()),indent=2))
                for i,row in enumerate(pending,1):
                    q=row['question_id']
                    values=capture(paths[q],meta['answer_only_prompt'],model,tok,features,decoder)
                    if values.shape!=base.shape or not np.isfinite(values).all():
                        raise ValueError(f'Invalid acoustic state: {q}')
                    write_state(states/(q+'.npy'),values)
                    print(f'[{len(rows)-len(pending)+i}/{len(rows)}] saved {q}',flush=True)
        report(folder,ref,rows)


if __name__=='__main__':
    main()
