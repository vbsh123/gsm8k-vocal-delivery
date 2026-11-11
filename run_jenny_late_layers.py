"""Sweep late-layer steering on the existing second-cohort neutral recordings."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from evaluate_audio_steering import report


LABEL = ('Second Azure cohort, same neutral recordings as the original timing run; '
         'late-layer development sweep, without outcome or classifier filtering.')


def conditions():
    result = [dict(name='baseline', interventions=[])]
    for layer in (29, 30, 31):
        for group in ('acoustic', 'answer'):
            for strength in (0.5, 1):
                for schedule in ('always', 'periodic2'):
                    name = f'L{layer}-{group}-a{strength}'
                    if schedule != 'always':
                        name += '-' + schedule
                    result.append(dict(name=name, schedule=schedule, interventions=[
                        dict(layer=layer, group=group, strength=strength)]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path,
                   default=Path('outputs/jenny-large/timing/original_mean'))
    p.add_argument('--vectors', type=Path, help='Override the original vector checkpoint path if moved.')
    p.add_argument('--output', type=Path, default=Path('outputs/steer-jenny-late-layers'))
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--summary-only', action='store_true')
    args = p.parse_args()
    if args.summary_only:
        report(args.output, ['original_mean'], LABEL)
        return
    source = json.loads((args.source_run / 'run_config.json').read_text())
    vector = args.vectors or Path(source['settings']['vectors'])
    if not vector.is_file():
        p.error(f'Missing original checkpoint: {vector}; pass --vectors if it moved.')
    args.output.mkdir(parents=True, exist_ok=True)
    file = args.output / 'conditions.json'
    file.write_text(json.dumps(conditions(), indent=2) + '\n')
    command = [sys.executable, '-u', 'evaluate_audio_steering.py',
               '--source-run', str(args.source_run), '--vectors', str(vector),
               '--methods', 'original_mean', '--conditions-file', str(file),
               '--num-examples', str(len(source['recordings'])),
               '--output', str(args.output), '--baseline-source', str(args.source_run),
               '--cohort-label', LABEL]
    print(f'{len(source["recordings"])} recordings × 24 interventions; '
          'compatible saved baselines are reused. Each condition runs separately, seed42.', flush=True)
    print('periodic2 uses forwards 1, 3, 5, ... within each decoding block.', flush=True)
    subprocess.run(command + ['--prepare-only'], check=True)
    if args.prepare_only:
        return
    with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
        print('Waiting for the shared GPU lock.', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        while subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            print('Waiting for active GPU work.', flush=True)
            time.sleep(30)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
