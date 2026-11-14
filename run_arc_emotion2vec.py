"""Classify ARC or CommonsenseQA recordings, then audit saved answers on emotion2vec-agreed pairs."""

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from steer import read_rows


def commonsense_reports(source,benchmark,steering):
    from evaluate_arc_audio import report
    output=benchmark/'emotion2vec-audit';output.mkdir(parents=True,exist_ok=True)
    models=[('DIFFA',read_rows(benchmark/'results.jsonl')),
        ('Qwen',read_rows(source/'results-qwen-commonsenseqa.jsonl')),
        ('Kimi',read_rows(source/'results-kimi-commonsenseqa.jsonl'))]
    for name,rows in models:
        if not rows:
            print(name+': no saved CommonsenseQA answers here.',flush=True);continue
        rows=[dict(r,condition=r['style']) for r in rows]
        for agreed in (False,True):
            suffix='-emotion2vec' if agreed else '-all'
            report(output,source,rows=rows,summary_name=name.lower()+suffix+'.json',
                emotion2vec_agreed=agreed,dataset_name='CommonsenseQA / '+name)
    saved=read_rows(steering/'results.jsonl')
    keys={(r['question_id'],r['voice']) for r in saved if r['condition']=='baseline'}
    if keys:
        matched=[dict(r,condition=r['style']) for r in models[0][1] if (r['question_id'],r['voice']) in keys]
        matched.extend(dict(r,condition='neutral-'+r['condition']) for r in saved if r['condition']!='baseline')
        for agreed in (False,True):
            suffix='-emotion2vec' if agreed else '-all'
            report(output,source,rows=matched,summary_name='diffa-heldout-steering'+suffix+'.json',
                emotion2vec_agreed=agreed,dataset_name='CommonsenseQA / DIFFA held-out steering cohort')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['arc','commonsenseqa'],default='arc')
    p.add_argument('--source',type=Path)
    p.add_argument('--benchmark',type=Path)
    p.add_argument('--arc-steering','--steering',dest='steering',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args(argv)
    common=args.task=='commonsenseqa'
    args.source=args.source or Path('outputs/azure-commonsenseqa500' if common else 'outputs/azure-arc-challenge500')
    args.benchmark=args.benchmark or Path('outputs/commonsenseqa-diffa2' if common else 'outputs/arc-challenge-diffa2')
    args.steering=args.steering or Path('outputs/commonsenseqa-angry-mean' if common else 'outputs/arc-challenge-angry-mean')
    if not (args.source/'plan.jsonl').exists():raise ValueError('Audio plan is missing: '+str(args.source))
    python=Path('.venv-emotion/bin/python').absolute()
    if not args.summary_only:
        if not python.exists():
            print('Set up the separate emotion environment; DIFFA environment is unchanged.',flush=True)
            subprocess.run(['bash','setup_emotion.sh'],env=dict(os.environ,PYTHON_BIN=sys.executable),check=True)
        with (args.source/'emotion2vec-queue.lock').open('w') as own:
            fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
                print('Waiting for the shared GPU lock; no DIFFA/Qwen/Kimi job is interrupted.',flush=True)
                fcntl.flock(lock,fcntl.LOCK_EX)
                if args.device.startswith('cuda'):
                    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                        print('Waiting for active GPU work.',flush=True);time.sleep(30)
                subprocess.run([str(python),'-u','classify_cloud_smoke.py','--path',str(args.source),
                    '--model','emotion2vec','--device',args.device],check=True)
    elif python.exists():
        subprocess.run([str(python),'classify_cloud_smoke.py','--path',str(args.source),
            '--model','emotion2vec','--summary-only'],check=True)
    if common:
        commonsense_reports(args.source,args.benchmark,args.steering)
        print('Only classification and saved-answer scoring ran; no accuracy generations or vector fitting.',flush=True)
        return
    print('\nALL INPUTS: primary paired delivery/transfer results',flush=True)
    subprocess.run([sys.executable,'evaluate_arc_audio.py','--source',str(args.source),
        '--output',str(args.benchmark),'--summary-only'],check=True)
    print('\nEMOTION2VEC AGREEMENT: each original input must match its requested style',flush=True)
    subprocess.run([sys.executable,'evaluate_arc_audio.py','--source',str(args.source),
        '--output',str(args.benchmark),'--summary-only','--emotion2vec-agreed'],check=True)
    if (args.steering/'split.json').exists():
        subprocess.run([sys.executable,'steer_arc_mean.py','--source',str(args.source),
            '--benchmark',str(args.benchmark),'--output',str(args.steering),
            '--summary-only','--emotion2vec-agreed'],check=True)
    print('No accuracy generations or vector fitting were performed; saved classifier labels resume.',flush=True)


if __name__=='__main__':main()
