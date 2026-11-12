"""Resume two L31 finalists on the current 460 recordings, then L12 forward20 periodic2."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from steer import read_rows


FINALISTS=['L31-answer-a1','L31-answer-a1-periodic2']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/steer-jenny-late-layers'))
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    original=args.root/'original_mean'
    temporal=args.root/'forward20'
    config=json.loads((original/'run_config.json').read_text())
    forward=json.loads((temporal/'run_config.json').read_text())
    saved={(r['clip_id'],r['condition']) for r in read_rows(original/'results.jsonl')}
    pending=sum((r['clip_id'],name) not in saved for r in config['recordings'] for name in FINALISTS)
    print(f'{pending} finalist generations pending; completed answers are retained. Other late-layer conditions stay partial.',flush=True)
    first=[sys.executable,'-u','steer_audio.py','--config',str(original/'run_config.json'),
           '--output',str(original),'--conditions',*FINALISTS]
    settings=forward['settings']
    conditions_file=args.root/'finalist-forward20-conditions.json'
    conditions_file.write_text(json.dumps(settings['conditions'],indent=2)+'\n')
    second=[sys.executable,'-u','evaluate_audio_steering.py','--source-run',settings['previous'],
            '--vectors',settings['vectors'],'--methods','forward20',
            '--conditions-file',str(conditions_file),
            '--num-examples',str(len(forward['recordings'])),'--output',str(args.root),
            '--baseline-source',str(original),'--cohort-label',settings['selection']]
    for command in (first,second):
        subprocess.run(command+['--prepare-only'],check=True)
    if args.prepare_only:
        return
    subprocess.run([sys.executable,'-u','wait_for_experiments.py','--kind','azure-emotions'],check=True)
    with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
        print('Waiting for the shared GPU lock. Stop the old full sweep to avoid finishing its remaining conditions.',flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            print('Waiting for active GPU work.',flush=True)
            time.sleep(30)
        subprocess.run(first,check=True)
        subprocess.run(second,check=True)
    print(f'Two L31 finalists and L12 forward20 periodic2 finished on {len(config["recordings"])} recordings. No additional cohort was queued.',flush=True)


if __name__=='__main__':
    main()
