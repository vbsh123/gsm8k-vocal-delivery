"""Resume original IFEval + MultiArith protocols, with copies outside outputs."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOTS = ('azure-spoken-ifeval', 'spoken-ifeval-diffa2', 'ifeval-angry-mean',
         'azure-multiarith580', 'multiarith580-angry-mean')
SUFFIXES = {'.json', '.jsonl', '.pt', '.wav', '.log'}


def snapshot(outputs, backup):
    """Copy completed files atomically; never delete anything in either location."""
    outputs, backup = outputs.resolve(), backup.resolve()
    if backup == outputs or outputs in backup.parents:
        raise ValueError('Backup directory must be outside outputs.')
    for name in ROOTS:
        source = outputs/name
        if not source.exists():
            continue
        for path in source.rglob('*'):
            if not path.is_file() or path.suffix not in SUFFIXES:
                continue
            target = backup/path.relative_to(outputs)
            info = path.stat()
            if target.exists():
                old = target.stat()
                if (old.st_size, old.st_mtime_ns) == (info.st_size, info.st_mtime_ns):
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name+'.copying')
            shutil.copy2(path, temporary)
            temporary.replace(target)


def run_stage(name, command, backup):
    print(f'\nSTAGE: {name}\nCommand: {command}', flush=True)
    snapshot(Path('outputs'), backup)
    child = subprocess.Popen(command, start_new_session=True)
    try:
        while True:
            try:
                code = child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                snapshot(Path('outputs'), backup)
                continue
            snapshot(Path('outputs'), backup)
            if code:
                raise subprocess.CalledProcessError(code, command)
            return
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['both', 'ifeval', 'multiarith'], default='both')
    parser.add_argument('--mode', choices=['all', 'summary'], default='all')
    parser.add_argument('--backup-root', type=Path, default=Path('recovery-backups'))
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    python = sys.executable
    if args.mode == 'summary':
        if args.task in ('both', 'ifeval'):
            subprocess.run([python, 'steer_ifeval_mean.py', '--mode', 'summary'], check=True)
        if args.task in ('both', 'multiarith'):
            subprocess.run([python, 'run_multiarith_mean.py', '--mode', 'summary'], check=True)
        return
    outputs, backup = Path('outputs').resolve(), args.backup_root.resolve()
    if backup == outputs or outputs in backup.parents:
        raise ValueError('Backup directory must be outside outputs.')
    args.backup_root.mkdir(parents=True, exist_ok=True)
    print(f'Backup copies: {args.backup_root.resolve()}; copy these off-machine too.', flush=True)
    with (args.backup_root/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot(Path('outputs'), args.backup_root)
        for kind in ('bbh-audio-models', 'reasoning-transfer', 'mean-steering', 'seven-object', 'strength-sweep'):
            run_stage(f'Wait for {kind}', [python, 'wait_for_experiments.py', '--kind', kind], args.backup_root)
        if args.task in ('both', 'multiarith'):
            run_stage('MultiArith original 290/290 collection and evaluation',
                      [python, 'run_multiarith_mean.py'], args.backup_root)
        if args.task in ('both', 'ifeval'):
            # Same full four-delivery TTS and initial evaluation as the deleted run.
            run_stage('IFEval four-style audio and checker dependencies',
                      ['bash', 'run_spoken_ifeval.sh', '--mode', 'generate'], args.backup_root)
            run_stage('IFEval original four-delivery baseline, 1024 tokens',
                      [python, 'run_spoken_ifeval.py', '--mode', 'evaluate'], args.backup_root)
            run_stage('IFEval original half collection / half steering',
                      [python, 'steer_ifeval_mean.py'], args.backup_root)
        print('Selected reruns complete. Raw outputs and separate backup copies retained.', flush=True)


if __name__ == '__main__':
    main()
