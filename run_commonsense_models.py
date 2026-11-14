"""CommonsenseQA Qwen, Kimi, then DIFFA extraction/held-out mean steering, one model at a time."""

import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import wave

from steer import read_rows

MODELS=[('qwen','qwen-omni','Qwen--Qwen2.5-Omni-7B','ae9e1690543ffd5c0221dc27f79834d0294cba00'),
        ('kimi','kimi-audio','moonshotai--Kimi-Audio-7B-Instruct','9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b')]


def cache_roots():
    return sorted({Path(p) for p in [os.environ.get('HF_HOME','outputs/hf-diffa'),'/workspace/hf-cache',
        '/workspace/.hf_home','outputs/hf-qwen-omni','outputs/hf-kimi-audio']})


def remove_weights(repo):
    for cache in cache_roots():
        folder=cache/'hub'/('models--'+repo)
        if folder.is_symlink():
            print('Remove model-cache link:',folder,flush=True);folder.unlink()
        elif folder.is_dir():
            print('Remove downloadable model weights only:',folder,flush=True);shutil.rmtree(folder)


def cached(repo,revision):
    folder=Path(os.environ['HF_HOME'])/'hub'/('models--'+repo)/'snapshots'/revision
    indexes=list(folder.glob('*.safetensors.index.json'))
    if indexes:
        names=set(json.loads(indexes[0].read_text())['weight_map'].values())
        return all((folder/n).is_file() for n in names)
    return (folder/'model.safetensors').is_file()


def pin_diffa_cache(home,path):
    if path.exists():return
    pins={}
    for repo in ('GSAI-ML/LLaDA-8B-Instruct','openai/whisper-large-v3','zhoujiaming777/DIFFA-2'):
        folder=home/'hub'/('models--'+repo.replace('/','--'))
        ref=folder/'refs/main'
        if not ref.exists():raise ValueError(f'Cannot release DIFFA weights without its original cached revision: {ref}')
        revision=ref.read_text().strip();snapshot=folder/'snapshots'/revision
        files=[str(p.relative_to(snapshot)) for p in snapshot.rglob('*') if p.is_file()]
        if not files:raise ValueError(f'No cached files for {repo}')
        pins[repo]=dict(revision=revision,files=files)
    path.write_text(json.dumps(pins,indent=2)+'\n')


def restore_diffa_cache(home,pins):
    from huggingface_hub import snapshot_download
    for repo,record in json.loads(pins.read_text()).items():
        snapshot_download(repo,revision=record['revision'],allow_patterns=record['files'],cache_dir=str(home/'hub'))
        ref=home/'hub'/('models--'+repo.replace('/','--'))/'refs/main'
        ref.parent.mkdir(parents=True,exist_ok=True);ref.write_text(record['revision'])


def completed(source,label,expected):
    return expected<={r['audio_path'] for r in read_rows(source/f'results-{label}-commonsenseqa.jsonl')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/azure-commonsenseqa500'))
    p.add_argument('--benchmark',type=Path,default=Path('outputs/commonsenseqa-diffa2'))
    p.add_argument('--output',type=Path,default=Path('outputs/commonsenseqa-followups'))
    p.add_argument('--skip-steering',action='store_true')
    p.add_argument('--release-diffa-cache',action='store_true',help='After previous DIFFA work exits, release LLaDA weights for Qwen/Kimi; restored for the final steering stage.')
    p.add_argument('--release-model-cache',action='store_true',help='Remove each Qwen/Kimi weight cache only after all its answers are saved.')
    p.add_argument('--qwen-python',type=Path,default=Path('/venv/qwen-omni/bin/python'))
    p.add_argument('--kimi-python',type=Path,default=Path('/venv/kimi-audio/bin/python'))
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    p.add_argument('--restore-diffa-only',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.restore_diffa_only:
        restore_diffa_cache(Path(os.environ['HF_HOME']),args.restore_diffa_only);return
    if args.summary_only:
        status=args.output/'status.json'
        if status.exists():print(status.read_text())
        if not args.skip_steering:
            subprocess.run([sys.executable,'steer_commonsense_mean.py','--summary-only'],check=True)
        for label,env,_,_ in MODELS:
            subprocess.run([str(getattr(args,label+'_python')),'evaluate_'+('qwen_omni' if label=='qwen' else 'kimi_audio')+'.py',
                '--task','commonsenseqa','--paths',str(args.source),'--num-questions','0','--styles','neutral','angry','happy','sad','--summary-only'],check=True)
        return
    expected=set()
    for row in read_rows(args.source/'plan.jsonl'):
        if row['provider']!='azure' or row['style'] not in ('neutral','angry','happy','sad'):continue
        with wave.open(str(args.source/row['audio_path']),'rb') as audio:
            if not 0<audio.getnframes()/audio.getframerate()<=30:raise ValueError('Expected every CommonsenseQA recording to fit 30 seconds')
        expected.add(row['audio_path'])
    if not expected:raise ValueError('No generated recordings')
    print(f'{len(expected)} CommonsenseQA recordings per model; all four deliveries; same A–E prompt/scorer as DIFFA.',flush=True)
    if args.prepare_only:
        if not args.skip_steering:subprocess.run([sys.executable,'steer_commonsense_mean.py','--benchmark',str(args.benchmark),'--prepare-only'],check=True)
        return
    with (args.output/'queue.lock').open('w') as own:
        try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('CommonsenseQA follow-up queue is already running')
        def stage(name,command):
            status=dict(stage=name,state='running',time=datetime.now(timezone.utc).isoformat())
            (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n');print('\nSTAGE:',name,flush=True)
            try:subprocess.run(command,check=True)
            except subprocess.CalledProcessError as error:
                status.update(state='failed',returncode=error.returncode)
                (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n');raise
        for kind in ('pitch-speed','commonsense'):
            stage('Wait for '+kind,[sys.executable,'-u','wait_for_experiments.py','--kind',kind])
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for the shared GPU lock.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
            home=Path(os.environ.get('HF_HOME') or next((str(p) for p in (Path('/workspace/hf-cache'),Path('/workspace/.hf_home')) if (p/'hub').is_dir()),'outputs/hf-diffa')).resolve()
            os.environ.update(HF_HOME=str(home),HF_HUB_CACHE=str(home/'hub'),HUGGINGFACE_HUB_CACHE=str(home/'hub'),PIP_NO_CACHE_DIR='1')
            for key in ('TRANSFORMERS_CACHE','PYTHONHOME','PYTHONPATH'):os.environ.pop(key,None)
            pins=args.output/'diffa-model-revisions.json'
            if args.release_diffa_cache:
                pin_diffa_cache(home,pins)
                remove_weights('GSAI-ML--LLaDA-8B-Instruct')
            for label,env,repo,revision in MODELS:
                python=getattr(args,label+'_python')
                if not completed(args.source,label,expected):
                    free=shutil.disk_usage('.').free/2**30;needed=2 if cached(repo,revision) else 25
                    print(f'{label}: free {free:.2f} GiB; need {needed} GiB including download reserve.',flush=True)
                    if free<needed:raise ValueError('Insufficient disk for the next model; results retained. Free space before resuming.')
                    if not python.exists():stage('Set up '+env,['bash','setup_'+('qwen_omni' if label=='qwen' else 'kimi_audio')+'.sh'])
                    script='evaluate_qwen_omni.py' if label=='qwen' else 'evaluate_kimi_audio.py'
                    stage('Evaluate '+label,[str(python),'-u',script,'--task','commonsenseqa','--paths',str(args.source),
                        '--num-questions','0','--styles','neutral','angry','happy','sad'])
                if not completed(args.source,label,expected):raise ValueError(f'{label} did not save every answer; its weights are retained')
                if args.release_model_cache:remove_weights(repo)
        if not args.skip_steering:
            if args.release_diffa_cache:
                stage('Restore the exact original DIFFA component revisions',
                    [sys.executable,'-u','run_commonsense_models.py','--restore-diffa-only',str(pins)])
                os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            stage('DIFFA: extract on 250 pairs; test L12 acoustic and L31 answer x1 on other 250',
                [sys.executable,'-u','steer_commonsense_mean.py','--benchmark',str(args.benchmark)])
        (args.output/'status.json').write_text(json.dumps(dict(stage='Qwen, Kimi, then DIFFA steering finished',state='finished'),indent=2)+'\n')


if __name__=='__main__':main()
