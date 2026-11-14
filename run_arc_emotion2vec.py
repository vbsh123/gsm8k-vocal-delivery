"""Classify ARC recordings with emotion2vec, then audit existing answers on agreed input pairs."""

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/azure-arc-challenge500'))
    p.add_argument('--benchmark',type=Path,default=Path('outputs/arc-challenge-diffa2'))
    p.add_argument('--arc-steering',type=Path,default=Path('outputs/arc-challenge-angry-mean'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    if not (args.source/'plan.jsonl').exists():raise ValueError('ARC audio plan is missing')
    python=Path('.venv-emotion/bin/python').resolve()
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
    print('\nALL INPUTS: primary paired delivery/transfer results',flush=True)
    subprocess.run([sys.executable,'evaluate_arc_audio.py','--source',str(args.source),
        '--output',str(args.benchmark),'--summary-only'],check=True)
    print('\nEMOTION2VEC AGREEMENT: each original input must match its requested style',flush=True)
    subprocess.run([sys.executable,'evaluate_arc_audio.py','--source',str(args.source),
        '--output',str(args.benchmark),'--summary-only','--emotion2vec-agreed'],check=True)
    if (args.arc_steering/'split.json').exists():
        subprocess.run([sys.executable,'steer_arc_mean.py','--source',str(args.source),
            '--benchmark',str(args.benchmark),'--output',str(args.arc_steering),
            '--summary-only','--emotion2vec-agreed'],check=True)
    print('No accuracy generations or vector fitting were performed; saved classifier labels resume.',flush=True)


if __name__=='__main__':main()
