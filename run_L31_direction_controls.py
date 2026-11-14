"""Compare the original L31 mean with its reverse and three norm-matched random directions."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from compare_cloud_accuracy import PROMPT
from extract_audio_vectors import save_tensor
from rescore_audio_steering import download,harness_filter,report
from steer import read_rows

METHODS=['reverse','random1','random2','random3']
CONDITIONS=[dict(name='baseline',interventions=[]),
    dict(name='L31-answer-a1',interventions=[dict(layer=31,group='answer',strength=1)])]


def control_directions(vector,torch):
    generator=torch.Generator(device='cpu').manual_seed(42)
    controls={'reverse':-vector}
    for name in METHODS[1:]:
        random=torch.randn(vector.shape,generator=generator,dtype=torch.float32)
        controls[name]=random*(vector.norm()/random.norm())
    return controls


def summarize(output,sources):
    Filter=harness_filter(output/'upstream')
    source=output/'upstream/gsm8k-test.jsonl'
    if not source.exists():source.write_bytes(download('https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'))
    bank=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    answers=[];seen=set()
    for cohort,folder in sources:
        config=json.loads((folder/'run_config.json').read_text());ids={r['question_id'] for r in config['recordings']}
        if ids&seen:raise ValueError('Source evaluation cohorts overlap')
        seen.update(ids)
        for row in read_rows(folder/'results.jsonl'):
            if row['condition']=='baseline':answers.append(row)
            elif row['condition']=='L31-answer-a1':answers.append(dict(row,condition='original-L31-a1'))
        for method in METHODS:
            for row in read_rows(output/'evaluation'/cohort/method/'results.jsonl'):
                if row['condition']=='L31-answer-a1':answers.append(dict(row,condition=method+'-L31-a1'))
    report(answers,'gsm8k',bank,output/'scoring-audit','L31 direction controls; completed pairs only',Filter)
    versus_original=[dict(r,condition='baseline') for r in answers if r['condition']=='original-L31-a1']
    versus_original.extend(r for r in answers if r['condition'] not in ('baseline','original-L31-a1'))
    report(versus_original,'gsm8k',bank,output/'versus-original','Controls versus original learned direction; negative gain favors original',Filter)
    print('All comparisons are exploratory and p-values are unadjusted.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--late-root',type=Path,default=Path('outputs/steer-jenny-late-layers'))
    p.add_argument('--output',type=Path,default=Path('outputs/jenny-L31-direction-controls'))
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sources=[('second',args.late_root/'original_mean'),('third',args.late_root/'third-cohort/original_mean')]
    if args.summary_only:summarize(args.output,sources);return
    import torch
    config=json.loads((sources[0][1]/'run_config.json').read_text())
    path=Path(config['settings']['vectors']);state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if state['prompt']!=PROMPT or state['max_new_tokens']!=512:raise ValueError('Expected original GSM8K vector recipe')
    index=state['groups'].index('answer');vector=(state['pair_mean_sums']['angry'][31,index]/state['pair_counts']['angry'][31]).float()
    controls=control_directions(vector,torch)
    recipe=dict(source_vector=str(path.resolve()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        methods=METHODS,construction_seed=42,inference_seed=42,layer=31,group='answer',strength=1,
        norm=float(vector.norm()),cosines={name:float(torch.nn.functional.cosine_similarity(vector,value,dim=0)) for name,value in controls.items()})
    frozen=args.output/'controls.json'
    if frozen.exists() and json.loads(frozen.read_text())!=recipe:raise ValueError('Frozen control vector recipe changed')
    frozen.write_text(json.dumps(recipe,indent=2)+'\n')
    for name,value in controls.items():
        folder=args.output/'vectors'/name;folder.mkdir(parents=True,exist_ok=True)
        checkpoint=folder/'style-vectors.pt'
        if checkpoint.exists():continue
        sums=torch.zeros_like(state['pair_mean_sums']['angry']);sums[31,index]=value
        counts=torch.zeros_like(state['pair_counts']['angry']);counts[31]=1
        save_tensor(torch,dict(pair_mean_sums={'angry':sums},pair_counts={'angry':counts},groups=state['groups'],
            prompt=PROMPT,max_new_tokens=512,mode='fixed',seed=42,method=name,collected_targets=[(31,'answer')],
            extraction_question_ids=state.get('extraction_question_ids',[])),checkpoint)
    del state,controls
    conditions=args.output/'conditions.json';conditions.write_text(json.dumps(CONDITIONS,indent=2)+'\n')
    commands=[];total=0
    for name,source in sources:
        count=len(json.loads((source/'run_config.json').read_text())['recordings']);total+=count
        command=[sys.executable,'-u','evaluate_audio_steering.py','--source-run',str(source),
            '--vectors-root',str(args.output/'vectors'),'--methods',*METHODS,
            '--conditions-file',str(conditions),'--num-examples',str(count),'--baseline-source',str(source),
            '--output',str(args.output/'evaluation'/name),'--cohort-label',
            'Original GSM8K evaluation recordings; L31 answer x1 reverse/random norm controls; seed42; development experiment.']
        subprocess.run(command+['--prepare-only'],check=True);commands.append(command)
    print(f'{total} recordings × 4 controls = {4*total} generations before cache reuse; existing baselines/original direction retained.',flush=True)
    print('Random directions use a separate CPU generator seeded42; inference always resets seed42.',flush=True)
    if args.prepare_only:return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for kind in ('finalist-queue','azure-emotions'):
            subprocess.run([sys.executable,'-u','wait_for_experiments.py','--kind',kind],check=True)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
            cache=os.environ.get('HF_HOME') or next((str(p) for p in (Path('/workspace/.hf_home'),Path('/workspace/hf-cache')) if (p/'hub').is_dir()),None)
            if cache:
                os.environ.update(HF_HOME=cache,HF_HUB_CACHE=str(Path(cache)/'hub'),HUGGINGFACE_HUB_CACHE=str(Path(cache)/'hub'))
                os.environ.pop('TRANSFORMERS_CACHE',None)
            for command in commands:subprocess.run(command,check=True)
    summarize(args.output,sources)


if __name__=='__main__':main()
