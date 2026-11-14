"""Learn same-state angry corrections, then test a mean and an adaptive L31 correction on held-out GSM8K."""

import argparse
import gc
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import fcntl

from activation_patch_audio import capture_prefix,patch_forward
from compare_cloud_accuracy import PROMPT
from evaluate_audio_steering import valid_baseline
from extract_audio_vectors import save_tensor
from scoring import score_response
from steer import read_rows,summarize

CONDITIONS=[dict(name='baseline'),dict(name='original-L31-a1'),dict(name='same-state-mean-L31-a1'),
            dict(name='adaptive-L31-a0.5'),dict(name='adaptive-L31-a1')]
TRIALS=[('same-state-mean-L31-a1',False,1.),('adaptive-L31-a0.5',True,.5),('adaptive-L31-a1',True,1.)]


def fit_mapping(rows,torch,rank=32,ridge=1.):
    # PCA uses question means; the regression gives each question equal total weight.
    xmean=torch.stack([r['neutral'].mean(0) for r in rows]).mean(0)
    dmean=torch.stack([r['difference'].mean(0) for r in rows]).mean(0)
    centers=torch.stack([r['neutral'].mean(0)-xmean for r in rows])
    _,_,vh=torch.linalg.svd(centers,full_matrices=False)
    rank=min(rank,len(vh),xmean.numel())
    basis=vh[:rank].T.contiguous()
    projected=[(r['neutral']-xmean)@basis for r in rows]
    scale=torch.stack([z.square().mean(0) for z in projected]).mean(0).sqrt().clamp_min(1e-6)
    normalized=[(z/scale).clamp(-5,5) for z in projected]
    zmean=torch.stack([z.mean(0) for z in normalized]).mean(0)
    cov=torch.zeros(rank,rank);cross=torch.zeros(rank,xmean.numel())
    for z,row in zip(normalized,rows):
        z=z-zmean;weight=1/(len(rows)*len(z))
        cov+=weight*z.T@z
        cross+=weight*z.T@(row['difference']-dmean)
    coef=torch.linalg.solve(cov+ridge*torch.eye(rank),cross)
    return dict(center=xmean,mean=dmean,basis=basis,scale=scale,feature_center=zmean,coef=coef,rank=rank,ridge=ridge)


def predict_raw(mapping,x):
    z=(((x-mapping['center'])@mapping['basis'])/mapping['scale']).clamp(-5,5)
    return mapping['mean']+(z-mapping['feature_center'])@mapping['coef']


def predict_delta(mapping,x,adaptive,strength):
    raw=predict_raw(mapping,x) if adaptive else mapping['mean']
    delta=raw*mapping['gain']
    factor=(2*mapping['reference_norm']/delta.norm().clamp_min(1e-12)).clamp(max=1)
    return strength*delta*factor


def validation(rows,torch):
    if len(rows)<5:return {}
    order=list(range(len(rows)));random.Random(42).shuffle(order);cut=int(.8*len(rows))
    mapping=fit_mapping([rows[i] for i in order[:cut]],torch)
    adaptive=[];constant=[]
    for i in order[cut:]:
        row=rows[i]
        adaptive.append(float((predict_raw(mapping,row['neutral'])-row['difference']).square().mean()))
        constant.append(float((mapping['mean']-row['difference']).square().mean()))
    return dict(validation_questions=len(adaptive),adaptive_mse=sum(adaptive)/len(adaptive),
                mean_mse=sum(constant)/len(constant))


def report(output,sources):
    all_rows=[];seen=set()
    for name,source in sources:
        config=json.loads((source/'run_config.json').read_text());ids={r['question_id'] for r in config['recordings']}
        if ids&seen:raise ValueError('Evaluation cohorts overlap')
        seen.update(ids)
        fresh=read_rows(output/'evaluation'/name/'results.jsonl');all_rows.extend(fresh)
        have={r['clip_id'] for r in fresh if r['condition']=='baseline'}
        for row in read_rows(source/'results.jsonl'):
            if row['condition']=='L31-answer-a1':all_rows.append(dict(row,condition='original-L31-a1'))
            elif row['condition']=='baseline' and row['clip_id'] not in have:all_rows.append(row)
    print('\nCOMBINED: existing original mean versus same-state mean and adaptive correction.',flush=True)
    result=summarize(all_rows,CONDITIONS)
    (output/'combined-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def collect(plan,output,torch):
    import librosa
    import numpy as np
    import soundfile as sf
    import src.fast_dllm.diffa_generate as decoder
    from extract_vectors import load_model,gen
    pending=[(i,p) for i,p in enumerate(plan['pairs']) if not (output/'pairs'/f'{i:05d}.pt').exists()]
    if not pending:return
    model,tokenizer,features=load_model();llm=model.llm_model
    layer=llm.model.transformer.blocks[31];bounds={};original=decoder.insert_audio_embedding
    def insert(text,speech,lengths):
        start=int(lengths[0].item());bounds.update(start=start,acoustic_end=start+model.config.prompt_size)
        return original(text,speech,lengths)
    decoder.insert_audio_embedding=insert
    def generate(audio):
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        return gen(audio,16000,model,tokenizer,features,PROMPT,512)
    try:
        for i,pair in pending:
            print(f'Collect [{i+1}/{len(plan["pairs"])}] {pair["question_id"]}, {pair["matched_forwards"]} matched-state forwards',flush=True)
            audio={}
            for style in ('neutral','angry'):
                record=pair[style];path=Path(record['audio_path']);stat=path.stat()
                if stat.st_size!=record['audio_bytes'] or stat.st_mtime_ns!=record['audio_mtime_ns']:raise ValueError('Training audio changed')
                a,sr=sf.read(path,dtype='float32')
                if a.ndim==2:a=a.mean(axis=1)
                if sr!=16000:a=librosa.resample(a,orig_sr=sr,target_sr=16000)
                audio[style]=a
            donor,db=capture_prefix(llm,lambda:generate(audio['angry']),bounds)
            recipient,rb=capture_prefix(llm,lambda:generate(audio['neutral']),bounds);del recipient
            reference=generate(audio['neutral']) if i<3 else None
            with patch_forward(llm,layer,donor,db,rb,'answer',torch,collect_only=True,max_forwards=pair['matched_forwards']) as state:
                response=generate(audio['neutral'])
            if reference is not None and response!=reference:raise ValueError('Non-mutating matched-state collection changed neutral generation')
            if len(state['neutral_means'])!=pair['matched_forwards']:raise ValueError('Collected fewer forwards than the frozen matched count')
            value=dict(question_id=pair['question_id'],neutral=torch.stack(state['neutral_means']),
                       difference=torch.stack(state['differences']),neutral_response=response,
                       matched_forwards=pair['matched_forwards'],recipient_forwards=state['recipient_forwards'])
            save_tensor(torch,value,output/'pairs'/f'{i:05d}.pt')
            del donor,state,value
    finally:decoder.insert_audio_embedding=original


def evaluate(plan,mapping,output,sources,torch):
    import librosa
    import numpy as np
    import soundfile as sf
    from extract_vectors import load_model,gen
    work=[]
    for name,source in sources:
        config=json.loads((source/'run_config.json').read_text());records=config['recordings']
        folder=output/'evaluation'/name;folder.mkdir(parents=True,exist_ok=True)
        frozen=dict(recordings=records,seed=42,prompt=PROMPT,trials=TRIALS)
        path=folder/'run_config.json'
        if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(frozen)):raise ValueError('Evaluation plan changed')
        path.write_text(json.dumps(frozen,indent=2)+'\n')
        rows=read_rows(folder/'results.jsonl',repair=True)
        if any(r.get('seed')!=42 or r.get('prompt')!=PROMPT for r in rows):raise ValueError('Cached evaluation recipe differs')
        complete={(r['clip_id'],r['condition']) for r in rows};expected={r['clip_id']:r for r in records}
        with (folder/'results.jsonl').open('a') as stream:
            for row in read_rows(source/'results.jsonl'):
                cid=row['clip_id']
                if cid in expected and valid_baseline(row,expected[cid]) and (cid,'baseline') not in complete:
                    stream.write(json.dumps(row)+'\n');complete.add((cid,'baseline'))
        if any((r['clip_id'],'baseline') not in complete for r in records):raise ValueError('Missing compatible baseline')
        work.extend((folder,r,trial) for r in records for trial in TRIALS if (r['clip_id'],trial[0]) not in complete)
    if not work:return
    mapping={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in mapping.items()}
    model,tokenizer,features=load_model();layer=model.llm_model.model.transformer.blocks[31]
    for index,(folder,record,(name,adaptive,strength)) in enumerate(work,1):
        print(f'Evaluate [{index}/{len(work)}] {record["question_id"]} {name}',flush=True)
        audio,sr=sf.read(record['audio_path'],dtype='float32')
        if audio.ndim==2:audio=audio.mean(axis=1)
        if sr!=16000:audio=librosa.resample(audio,orig_sr=sr,target_sr=16000)
        state=dict(forwards=0)
        def hook(module,args,result):
            hidden=result[0].clone();x=hidden[0,-512:].float().mean(0)
            delta=predict_delta(mapping,x,adaptive,strength)
            hidden[:,-512:]+=delta.to(dtype=hidden.dtype);state['forwards']+=1
            return (hidden,*result[1:])
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        handle=layer.register_forward_hook(hook);started=time.monotonic()
        try:response=gen(audio,16000,model,tokenizer,features,PROMPT,512)
        finally:handle.remove()
        row=dict(clip_id=record['clip_id'],question_id=record['question_id'],condition=name,seed=42,prompt=PROMPT,
                 response=response,forwards=state['forwards'],generation_seconds=time.monotonic()-started,
                 **score_response(response,record['answer']))
        with (folder/'results.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        print(f'  correct={row["correct"]} answer={row["predicted_answer"]}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,default=Path('outputs/jenny-svd-probe'))
    p.add_argument('--late-root',type=Path,default=Path('outputs/steer-jenny-late-layers'))
    p.add_argument('--output',type=Path,default=Path('outputs/jenny-L31-adaptive'))
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sources=[('second',args.late_root/'original_mean'),('third',args.late_root/'third-cohort/original_mean')]
    if args.summary_only:report(args.output,sources);return
    import torch
    torch.set_num_threads(min(8,torch.get_num_threads()));torch.manual_seed(42)
    original=json.loads((sources[0][1]/'run_config.json').read_text())
    vector=Path(original['settings']['vectors']);data=torch.load(vector,map_location='cpu',weights_only=True,mmap=True)
    refnorm=float((data['pair_mean_sums']['angry'][31,data['groups'].index('answer')]/data['pair_counts']['angry'][31]).norm());del data
    source=json.loads((args.collection/'collection-config.json').read_text())
    if source['prompt']!=PROMPT or source.get('averaging','matched')!='matched':p.error('Expected original matched-forward GSM8K collection')
    pairs=[]
    for i,pair in enumerate(source['pairs']):
        old=torch.load(args.collection/'pairs'/f'{i:05d}.pt',map_location='cpu',weights_only=True)
        count=old.get('matched_forwards')
        if not count:p.error('Original pair cache has no matched-forward count')
        row=dict(question_id=pair['question_id'],matched_forwards=count)
        for style in ('neutral','angry'):
            record=source['manifest'][pair[style+'_id']];path=Path(record['audio_path']);stat=path.stat()
            row[style]=dict(audio_path=str(path),audio_bytes=stat.st_size,audio_mtime_ns=stat.st_mtime_ns)
        pairs.append(row)
    train={r['question_id'] for r in pairs}
    for _,folder in sources:
        if train&{r['question_id'] for r in json.loads((folder/'run_config.json').read_text())['recordings']}:p.error('Training/evaluation questions overlap')
    plan=dict(pairs=pairs,prompt=PROMPT,seed=42,rank=32,ridge=1.,reference_norm=refnorm,
              reference_vector=str(vector.resolve()),reference_bytes=vector.stat().st_size,
              reference_mtime_ns=vector.stat().st_mtime_ns,feature_clip=5.,correction_norm_cap=2.)
    config=args.output/'collection-config.json'
    if config.exists() and json.loads(config.read_text())!=plan:p.error('Adaptive extraction plan changed; use another output')
    config.write_text(json.dumps(plan,indent=2)+'\n');(args.output/'pairs').mkdir(exist_ok=True)
    print(f'{len(pairs)} training pairs; same answer state and old matched-forward counts; adaptive rank32/ridge1.',flush=True)
    print('Held-out evaluation: same-state mean x1, adaptive x0.5/x1; no angry recordings at test time.',flush=True)
    if args.prepare_only:return
    with (args.output/'queue.lock').open('w') as own:
        try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Adaptive queue is already running')
        for kind in ('finalist-queue','azure-emotions'):
            subprocess.run([sys.executable,'-u','wait_for_experiments.py','--kind',kind],check=True)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for the shared GPU lock.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
            cache=next((root for root in (Path('/workspace/.hf_home'),Path('/workspace/hf-cache')) if (root/'hub').is_dir()),None)
            if cache:
                import os
                os.environ.setdefault('HF_HOME',str(cache))
                os.environ['HF_HUB_CACHE']=str(Path(os.environ['HF_HOME'])/'hub');os.environ['HUGGINGFACE_HUB_CACHE']=os.environ['HF_HUB_CACHE']
                os.environ.pop('TRANSFORMERS_CACHE',None)
            def status(stage):
                (args.output/'status.json').write_text(json.dumps(dict(stage=stage),indent=2)+'\n');print('\nSTAGE:',stage,flush=True)
            status('Collect same-state neutral features and angry corrections')
            collect(plan,args.output,torch);gc.collect();torch.cuda.empty_cache()
            if torch.cuda.memory_allocated()>2*2**30:raise RuntimeError('Collector still holds GPU tensors; refusing a second model load')
            checkpoint=args.output/'predictor.pt'
            if checkpoint.exists():mapping=torch.load(checkpoint,map_location='cpu',weights_only=True)
            else:
                status('Fit regularized predictor on extraction cohort only')
                rows=[torch.load(args.output/'pairs'/f'{i:05d}.pt',map_location='cpu',weights_only=True) for i in range(len(pairs))]
                metrics=validation(rows,torch);mapping=fit_mapping(rows,torch)
                if mapping['mean'].norm()<=1e-12:raise ValueError('Same-state mean has zero norm')
                mapping.update(reference_norm=refnorm,gain=refnorm/float(mapping['mean'].norm()),validation=metrics,
                               seed=42,prompt=PROMPT,extraction_question_ids=sorted(train),layer=31,group='answer')
                save_tensor(torch,mapping,checkpoint)
                (args.output/'fit-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print('Activation-prediction validation:',metrics,flush=True)
                del rows
            status('Evaluate three interventions on both held-out cohorts')
            evaluate(plan,mapping,args.output,sources,torch)
            report(args.output,sources);status('Finished')


if __name__=='__main__':main()
