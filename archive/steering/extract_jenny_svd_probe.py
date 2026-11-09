"""Collect paired activations once; fit per-layer SVD and linear-head directions.

Rows are question-level angry-minus-neutral differences, averaged over matched
forwards. Fits mean, uncentered SVD rank1, projected mean, and a linear head.
All fitted directions have the raw mean's norm. No evaluation answers are used.
"""

import argparse
import gc
import json
from pathlib import Path
import random

from compare_cloud_accuracy import PROMPT
from archive.steering.extract_jenny_angry import GROUPS, recordings, select_pairs


def save_tensor(torch, value, path):
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def collect(args, pairs, manifest, torch, np, sf):
    folder=args.output/'pairs'
    folder.mkdir(exist_ok=True)
    pending=[(i,p) for i,p in enumerate(pairs) if not (folder/f'{i:05d}.pt').exists()]
    print(f'{len(pairs)} pairs; {len(pending)*2} generations pending; seed always42.',flush=True)
    if not pending:
        return
    import src.fast_dllm.diffa_generate as decoder
    from extract_vectors import load_model,gen
    model,tokenizer,features=load_model()
    layers=model.llm_model.model.transformer.blocks
    bounds={}
    captured=[]
    original=decoder.insert_audio_embedding

    def insert(text_embeds,speech_embeds,audio_info_lengths):
        start=int(audio_info_lengths[0].item())
        bounds.update(start=start,acoustic_end=start+model.config.prompt_size,end=start+speech_embeds.shape[1])
        return original(text_embeds,speech_embeds,audio_info_lengths)

    def hook_for(layer):
        def hook(module,inputs,output):
            h=output[0]
            positions=[slice(bounds['start'],bounds['acoustic_end']),slice(bounds['acoustic_end'],bounds['end']),
                       slice(bounds['start'],bounds['end']),slice(-512,None)]
            captured[layer].append(torch.stack([h[0,p].float().mean(0) for p in positions]).cpu())
        return hook

    def capture(clip):
        nonlocal captured
        captured=[[] for _ in layers]
        bounds.clear()
        audio,rate=sf.read(manifest[clip]['audio_path'],dtype='float32')
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        response=gen(audio,rate,model,tokenizer,features,PROMPT,512)
        states=torch.stack([torch.stack(passes) for passes in captured])
        captured=[]
        return response,states

    handles=[layer.register_forward_hook(hook_for(i)) for i,layer in enumerate(layers)]
    decoder.insert_audio_embedding=insert
    try:
        for i,pair in pending:
            print(f'[{i+1}/{len(pairs)}] {pair["question_id"]} neutral + angry',flush=True)
            nr,n=capture(pair['neutral_id'])
            ar,a=capture(pair['angry_id'])
            steps=min(n.shape[1],a.shape[1])
            row=dict(pair=pair,neutral=n[:,:steps].mean(1),angry=a[:,:steps].mean(1),
                     neutral_response=nr,angry_response=ar,matched_forwards=steps)
            save_tensor(torch,row,folder/f'{i:05d}.pt')
            print(f'  matched forwards={steps}; saved question-level means.',flush=True)
            del n,a,row
    finally:
        for handle in handles:
            handle.remove()
        decoder.insert_audio_embedding=original
    del model,layers
    gc.collect()
    torch.cuda.empty_cache()


def fit_one(d, rank, iterations, regularization, torch):
    mean=d.mean(0)
    _,singular,vh=torch.linalg.svd(d,full_matrices=False)
    top=vh[0]
    if torch.dot(top,mean)<0:
        top=-top
    basis=vh[:min(rank,len(vh))]
    projected=(mean@basis.T)@basis
    # A paired midpoint removes shared question content: neutral=-d/2, angry=+d/2.
    x=torch.cat([-d/2,d/2])
    scale=x.square().mean().sqrt().clamp_min(1e-8)
    x=x/scale
    y=torch.cat([torch.zeros(len(d)),torch.ones(len(d))])
    w=torch.zeros(d.shape[1],requires_grad=True)
    bias=torch.zeros((),requires_grad=True)
    optimizer=torch.optim.LBFGS([w,bias],max_iter=iterations,line_search_fn='strong_wolfe')
    def loss():
        optimizer.zero_grad()
        value=torch.nn.functional.binary_cross_entropy_with_logits(x@w+bias,y)+regularization*w.square().sum()/2
        value.backward()
        return value
    optimizer.step(loss)
    probe=w.detach()/scale
    result={'mean':mean,'svd1':top,'svd_mean':projected,'probe':probe}
    for name,v in result.items():
        if name!='mean':
            result[name]=v/v.norm().clamp_min(1e-12)*mean.norm()
    energy=singular.square().sum().clamp_min(1e-12)
    metrics=dict(rank=min(rank,len(vh)),energy_rank1=float(singular[0].square()/energy),
                 energy_topk=float(singular[:rank].square().sum()/energy),
                 probe_training_accuracy=float(((x@w.detach()+bias.detach()>0)==y.bool()).float().mean()),
                 cosine_to_mean={name:float(torch.nn.functional.cosine_similarity(v,mean,dim=0)) for name,v in result.items()})
    return result,metrics


def fit(args,pairs,excluded,torch):
    torch.manual_seed(42)
    torch.set_num_threads(min(8,torch.get_num_threads()))
    values=[torch.load(args.output/'pairs'/f'{i:05d}.pt',map_location='cpu',weights_only=True) for i in range(len(pairs))]
    neutral=torch.stack([v['neutral'] for v in values])
    angry=torch.stack([v['angry'] for v in values])
    del values
    difference=angry-neutral
    del neutral,angry
    n,layers,groups,width=difference.shape
    methods={name:torch.zeros(layers,groups,width) for name in ['mean','svd1','svd_mean','probe']}
    metrics=[]
    for layer in range(layers):
        for group in range(groups):
            vectors,info=fit_one(difference[:,layer,group],args.rank,args.probe_iterations,args.l2,torch)
            for name,v in vectors.items():
                methods[name][layer,group]=v
            metrics.append(dict(layer=layer,group=GROUPS[group],**info))
            print(f'Fit L{layer} {GROUPS[group]}: top1 energy={info["energy_rank1"]:.1%}; probe train accuracy={info["probe_training_accuracy"]:.1%}',flush=True)
    for name,vectors in methods.items():
        folder=args.output/name
        folder.mkdir(exist_ok=True)
        state=dict(pair_mean_sums={'angry':vectors*n},pair_counts={'angry':torch.full((layers,),n,dtype=torch.long)},
                   groups=GROUPS,styles=['angry'],prompt=PROMPT,max_new_tokens=512,steps=128,
                   sign='style minus neutral',seed=42,seed_policy='constant_before_every_generation',
                   extraction_question_ids=excluded,pairs=pairs,method=name,supported_modes=['fixed'],
                   classifier='emotion2vec',norm='matched to raw mean per layer/group',
                   rank=args.rank,probe_l2=args.l2,probe_iterations=args.probe_iterations)
        save_tensor(torch,state,folder/'style-vectors.pt')
        (folder/'extraction-questions.jsonl').write_text(''.join(json.dumps(dict(question_id=q))+'\n' for q in excluded))
        print('Saved:',folder/'style-vectors.pt',flush=True)
    (args.output/'fit-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',type=Path,nargs='+',default=[Path('outputs/azure-jenny500-extra'),Path('outputs/azure-jenny500-neutral')])
    p.add_argument('--output',type=Path,default=Path('outputs/jenny-svd-probe'))
    p.add_argument('--rank',type=int,default=5)
    p.add_argument('--probe-iterations',type=int,default=50)
    p.add_argument('--l2',type=float,default=.01)
    p.add_argument('--limit',type=int)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--fit-only',action='store_true',help='Refit directions from saved question means without loading DIFFA.')
    args=p.parse_args()
    records,labels=recordings(args.paths)
    pairs=select_pairs(records,labels,42)
    if args.limit:
        pairs=pairs[:args.limit]
    if not pairs or args.rank<1 or args.probe_iterations<1 or args.l2<=0:
        p.error('Need matched pairs, positive rank/iterations and positive L2')
    manifest={r['clip_id']:r for r in records.values()}
    excluded=sorted({r['question_id'] for r in records.values()})
    plan=dict(pairs=pairs,manifest=manifest,seed=42,prompt=PROMPT,max_new_tokens=512,groups=GROUPS)
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'collection-config.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        p.error('Collection changed; use another output folder')
    path.write_text(json.dumps(plan,indent=2)+'\n')
    print(f'{len(pairs)} pairs; one shared collection for mean, SVD and linear-head methods.',flush=True)
    if args.prepare_only:
        return
    import numpy as np
    import soundfile as sf
    import torch
    if not args.fit_only:
        collect(args,pairs,manifest,torch,np,sf)
    fit(args,pairs,excluded,torch)


if __name__=='__main__':
    main()
