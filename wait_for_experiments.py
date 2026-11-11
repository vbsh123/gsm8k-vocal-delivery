"""Wait for classification or steering processes in this repository to exit."""

import argparse
import os
from pathlib import Path
import time


NAMES={
    'classification':{'classify_dataset_emotions.py','classify_cloud_smoke.py',
                      'run_dataset_emotions.sh','run_speeds_and_classification.sh'},
    'steering':{'run_jenny_promoted_third.py','evaluate_audio_steering.py','steer_audio.py',
                'run_large_steering.py','run_large_steering.sh'},
}


def active_jobs(kind, root, proc=Path('/proc')):
    active=[]
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():
            continue
        try:
            if (entry/'cwd').resolve()!=root.resolve():
                continue
            state=(entry/'stat').read_text().rsplit(')',1)[1].split()[0]
            if state in ('Z','X'):
                continue
            args=[a.decode(errors='replace') for a in (entry/'cmdline').read_bytes().split(b'\0') if a]
            names={Path(a).name for a in args}
            matches=bool(names & NAMES[kind])
            if 'run_azure_followups.py' in names and '--stage' in args:
                matches |= args[args.index('--stage')+1] == ('classify' if kind=='classification' else 'steering')
            if matches:
                active.append(int(entry.name))
        except (OSError,IndexError):
            continue
    return sorted(active)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',required=True,choices=NAMES)
    p.add_argument('--poll-seconds',type=float,default=30)
    args=p.parse_args()
    if args.poll_seconds<=0:
        p.error('--poll-seconds must be positive')
    while True:
        pids=active_jobs(args.kind,Path.cwd())
        if not pids:
            print(f'No active {args.kind} jobs in this repository; continuing.',flush=True)
            return
        print(f'Waiting for {args.kind} PID(s): {pids}',flush=True)
        time.sleep(args.poll_seconds)


if __name__=='__main__':
    main()
