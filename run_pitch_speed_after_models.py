"""Wait for Qwen/Kimi, then DIFFA pitch controls and expanded speed evaluation."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import wave
import zipfile

from compare_cloud_accuracy import PROMPT
from steer import read_rows

RATES=[1.4,1.5,1.1,.9,.8,.7]
AZURE=[Path('outputs/azure-jenny-next500'),Path('outputs/azure-jenny-remaining-test')]


def complete_model_results():
    for folder in AZURE:
        eligible=set()
        for task in read_rows(folder/'plan.jsonl'):
            with wave.open(str(folder/task['audio_path']),'rb') as audio:
                if 0<audio.getnframes()/audio.getframerate()<=30:
                    eligible.add(task['audio_path'])
        for model in ('qwen-omni','kimi-audio'):
            saved={r['audio_path'] for r in read_rows(folder/f'results-{model}.jsonl')}
            missing=eligible-saved
            if missing:
                raise ValueError(f'{folder}: {model} still has {len(missing)} eligible clips without saved answers; model caches retained.')


def release_model_weights():
    complete_model_results()
    for model,default in [('Qwen--Qwen2.5-Omni-7B','outputs/hf-qwen-omni'),
                          ('moonshotai--Kimi-Audio-7B-Instruct','outputs/hf-kimi-audio')]:
        caches={Path(os.environ.get('HF_HOME') or default),Path(default)}
        for cache in sorted(caches):
            folder=cache/'hub'/f'models--{model}'
            if folder.is_dir():
                print(f'Removing downloadable model cache only: {folder}; results and environments retained.',flush=True)
                shutil.rmtree(folder)


def import_metadata(archive):
    allowed={f'outputs/azure-jenny-remaining-test/{name}' for name in (
        'results-matched-accuracy.jsonl','results-emotion2vec.jsonl','results-sensevoice.jsonl')}
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            if info.filename not in allowed:
                raise ValueError(f'Unexpected metadata member: {info.filename}')
            path=Path(info.filename)
            if not path.exists():
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(z.read(info))
                print('Imported:',path,flush=True)


def keep_samples_and_clear_cache(folder):
    audio=folder/'audio'
    if not audio.is_dir():
        return
    config=json.loads((folder/'run_config.json').read_text())
    samples=folder/'listening-samples';samples.mkdir(exist_ok=True)
    sample_ids={r['clip_id'] for r in config['recordings'][:3]}
    with (folder/'rendered-audio-sha256.jsonl').open('w') as stream:
        for file in sorted(audio.glob('*.wav')):
            h=hashlib.sha256()
            with file.open('rb') as source:
                for chunk in iter(lambda:source.read(1024**2),b''):
                    h.update(chunk)
            stream.write(json.dumps(dict(audio_path=file.name,bytes=file.stat().st_size,sha256=h.hexdigest()))+'\n')
            if any(file.name.startswith(cid+'__') for cid in sample_ids):
                shutil.copy2(file,samples/file.name)
    shutil.rmtree(audio)
    print(f'Retained results, waveform hashes and three listening examples; removed generated speed cache: {audio}',flush=True)


def speed_complete(folder):
    config=folder/'run_config.json'
    if not config.exists():
        return False
    plan=json.loads(config.read_text())
    saved={(r['clip_id'],r['condition']) for r in read_rows(folder/'results.jsonl')}
    return all((r['clip_id'],c['name']) in saved for r in plan['recordings'] for c in plan['conditions'])


def speed_report(source,output):
    import numpy as np
    all_rows={}
    for folder in [source,*[output/'speeds'/f'rate-{r:g}' for r in RATES]]:
        for row in read_rows(folder/'results.jsonl'):
            if row.get('seed')!=42 or row.get('prompt')!=PROMPT:
                raise ValueError('Different speed answer recipe')
            key=row['clip_id'],row['rate']
            if key in all_rows and (all_rows[key]['response'],all_rows[key]['correct'])!=(row['response'],row['correct']):
                raise ValueError(f'Conflicting speed answers: {key}')
            all_rows[key]=row
    baseline={cid:r for (cid,rate),r in all_rows.items() if rate==1}
    rates=sorted({rate for cid,rate in all_rows})
    common=set(baseline)
    for rate in rates:
        common &= {cid for cid,r in all_rows if r==rate}
    summary={}
    for title,ids in [('All eligible matched pairs at each rate',set(baseline)),
                      ('Same questions at every available rate',common)]:
        print(f'\n{title}; available rates: {rates}',flush=True)
        print(f'{"Rate":>5} {"Pairs":>6} {"Baseline":>9} {"Accuracy":>9} {"Gain pp":>8} {"Wins":>5} {"Losses":>6} {"Exact p":>9}   95% CI gain pp')
        values=[]
        for rate in rates:
            pairs=[(baseline[cid],all_rows[cid,rate]) for cid in sorted(ids) if (cid,rate) in all_rows]
            if not pairs:
                continue
            n=len(pairs);a=sum(b['correct'] for b,t in pairs);b=sum(t['correct'] for base,t in pairs)
            w=sum(t['correct'] and not base['correct'] for base,t in pairs)
            l=sum(base['correct'] and not t['correct'] for base,t in pairs)
            d=w+l;p=min(1,2*sum(math.comb(d,k) for k in range(min(w,l)+1))/2**d)
            draws=np.random.default_rng(42).multinomial(n,[w/n,l/n,(n-d)/n],size=200000)
            ci=np.percentile(100*(draws[:,0]-draws[:,1])/n,[2.5,97.5]).tolist()
            values.append(dict(rate=rate,pairs=n,baseline_correct=a,correct=b,wins=w,losses=l,mcnemar_exact_p=p,ci95_gain_pp=ci))
            print(f'{rate:5g} {n:6} {a/n:9.1%} {b/n:9.1%} {100*(b-a)/n:+8.2f} {w:5} {l:6} {p:9.4f}   [{ci[0]:+.1f}, {ci[1]:+.1f}]')
        summary[title]=values
    (output/'speed-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-speed-run',type=Path,default=Path('outputs/spoken-gsm-audiostretchy-remaining'))
    p.add_argument('--output',type=Path,default=Path('outputs/pitch-speed-night'))
    p.add_argument('--release-model-cache',action='store_true',help='After both model evaluations finish, remove only Qwen/Kimi downloadable weights to fit DIFFA.')
    p.add_argument('--keep-speed-audio',action='store_true',help='Keep all newly rendered speed WAVs; requires more disk.')
    p.add_argument('--metadata-zip',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.summary_only:
        status=args.output/'status.json'
        if status.exists():print(status.read_text())
        pitch=args.output/'pitch'
        if (pitch/'run_config.json').exists():
            from evaluate_audio_pitch import report
            report(pitch)
        speed_report(args.source_speed_run,args.output)
        return
    source=json.loads((args.source_speed_run/'run_config.json').read_text())
    for record in source['recordings']:
        if not Path(record['audio_path']).is_file():
            p.error(f'Missing original audio: {record["audio_path"]}')
    for folder in AZURE:
        for task in read_rows(folder/'plan.jsonl'):
            if not (folder/task['audio_path']).is_file():
                p.error(f'Missing Azure recording: {folder/task["audio_path"]}')
    print('Speed additions: 1.4/1.5/1.1 on all 1054 source questions; slower rates use duration-eligible subsets, no cropping.',flush=True)
    for rate in RATES:
        count=sum(r['duration_seconds']<=30*min(1,rate) for r in source['recordings'])
        print(f'  {rate:g}x: {count} eligible source recordings.',flush=True)
    if args.prepare_only:return
    with (args.output/'queue.lock').open('w') as own:
        try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Pitch/speed queue is already running')
        def stage(name,command):
            print(f'\nSTAGE: {name}',flush=True)
            status=dict(stage=name,state='running',time=datetime.now(timezone.utc).isoformat())
            (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
            try:subprocess.run(command,check=True)
            except subprocess.CalledProcessError as error:
                status.update(state='failed',returncode=error.returncode)
                (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n');raise
        stage('Wait for Qwen/Kimi queue and workers',[sys.executable,'-u','wait_for_experiments.py','--kind','audio-models'])
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
            if not os.environ.get('HF_HOME') and Path('/workspace/.hf_home').is_dir():
                os.environ['HF_HOME']='/workspace/.hf_home'
            if args.release_model_cache:release_model_weights()
            if os.environ.get('HF_HOME'):
                os.environ['HF_HUB_CACHE']=str(Path(os.environ['HF_HOME'])/'hub')
                os.environ['HUGGINGFACE_HUB_CACHE']=os.environ['HF_HUB_CACHE']
            for key in ('PYTHONHOME','PYTHONPATH','TRANSFORMERS_CACHE'):os.environ.pop(key,None)
            free=shutil.disk_usage('.').free/2**30
            print(f'Free disk after optional model-cache release: {free:.3f} GiB.',flush=True)
            if free<20:
                raise ValueError('Need at least 20 GiB free for restored DIFFA weights and temporary inputs; experiment audio/results have not been removed.')
            python=str(Path('.venv-diffa/bin/python').absolute())
            stage('Ensure CPU transform dependencies',[python,'-m','pip','install','--no-cache-dir','audiostretchy==1.3.14','imageio-ffmpeg==0.8.1'])
            archive=args.metadata_zip or next((p for p in (Path('/workspace/azure-final-metadata.zip'),Path('azure-final-metadata.zip')) if p.is_file()),None)
            if archive:import_metadata(archive)
            else:print('No optional final-cohort metadata ZIP; original pitch-control answers will be generated where absent.',flush=True)
            stage('Pitch controls: measure, render, audit and evaluate',[python,'-u','evaluate_audio_pitch.py','--output',str(args.output/'pitch')])
            for rate in RATES:
                folder=args.output/'speeds'/f'rate-{rate:g}'
                if not speed_complete(folder):
                    stage(f'Speed {rate:g}x', [python,'-u','evaluate_audio_speed.py','--source-run',str(args.source_speed_run),
                        '--num-examples','0','--rates','1',str(rate),'--stretch-method','audiostretchy',
                        '--reuse-from',str(args.source_speed_run),'--output',str(folder),'--save-audio'])
                if not args.keep_speed_audio:keep_samples_and_clear_cache(folder)
                speed_report(args.source_speed_run,args.output)
        (args.output/'status.json').write_text(json.dumps(dict(stage='Pitch and requested speed runs complete',state='finished',time=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')


if __name__=='__main__':
    main()
