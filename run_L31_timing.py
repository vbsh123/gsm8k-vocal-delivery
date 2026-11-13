"""Evaluate original-mean L31 answer x1 in early/middle/late decoding phases on both cohorts."""

import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from steer import read_rows,summarize


def conditions():
    return [dict(name='baseline',interventions=[])]+[
        dict(name=f'L31-answer-a1-{phase}',schedule=phase,
             interventions=[dict(layer=31,group='answer',strength=1)])
        for phase in ('early','middle','late')]


def report(output,sources):
    answers=[];seen=set();planned=0
    for name,source in sources:
        path=source/'run_config.json'
        config=json.loads(path.read_text())
        ids={r['question_id'] for r in config['recordings']}
        if ids&seen:raise ValueError('Cannot pool overlapping cohorts')
        seen.update(ids);planned+=len(config['recordings'])
        new=read_rows(output/name/'original_mean/results.jsonl')
        answers.extend(new)
        saved={r['clip_id'] for r in new if r['condition']=='baseline'}
        answers.extend(r for r in read_rows(source/'results.jsonl')
                       if r['condition'] in ('L31-answer-a1','L31-answer-a1-periodic2')
                       or (r['condition']=='baseline' and r['clip_id'] not in saved))
    comparison=[dict(name='baseline',interventions=[]),
                dict(name='L31-answer-a1',interventions=[dict(layer=31,group='answer',strength=1)]),
                dict(name='L31-answer-a1-periodic2',schedule='periodic2',interventions=[dict(layer=31,group='answer',strength=1)]),
                *conditions()[1:]]
    print(f'\nCOMBINED L31 timing: {planned} planned recordings; completed pairs only.',flush=True)
    rows=summarize(answers,comparison)
    (output/'combined-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--late-root',type=Path,default=Path('outputs/steer-jenny-late-layers'))
    p.add_argument('--output',type=Path,default=Path('outputs/jenny-L31-timing'))
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    sources=[('second',args.late_root/'original_mean'),('third',args.late_root/'third-cohort/original_mean')]
    args.output.mkdir(parents=True,exist_ok=True)
    if args.summary_only:
        report(args.output,sources);return
    config=json.loads((sources[0][1]/'run_config.json').read_text())
    vector=Path(config['settings']['vectors'])
    specs=args.output/'conditions.json';specs.write_text(json.dumps(conditions(),indent=2)+'\n')
    commands=[];count=0
    for name,source in sources:
        records=json.loads((source/'run_config.json').read_text())['recordings'];count+=len(records)
        command=[sys.executable,'-u','evaluate_audio_steering.py','--source-run',str(source),
            '--vectors',str(vector),'--methods','original_mean','--conditions-file',str(specs),
            '--num-examples',str(len(records)),'--output',str(args.output/name),
            '--baseline-source',str(source),'--cohort-label',
            f'L31 answer x1 original mean; early/middle/late thirds of unmasking within each 32-token block; {name} cohort.']
        subprocess.run(command+['--prepare-only'],check=True);commands.append((name,command))
    print(f'{count} recordings x 3 schedules = {3*count} intervention answers before cached results; compatible seed42 baselines reused.',flush=True)
    print('No extraction; the winning original-mean vector is unchanged. Always/periodic2 are shown from saved results.',flush=True)
    if args.prepare_only:return
    with (args.output/'queue.lock').open('w') as own:
        try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('L31 timing queue is already running')
        for kind in ('finalist-queue','azure-emotions'):
            subprocess.run([sys.executable,'-u','wait_for_experiments.py','--kind',kind],check=True)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for the shared GPU lock.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True);time.sleep(30)
            for name,command in commands:
                status=dict(stage=name,state='running',time=datetime.now(timezone.utc).isoformat())
                (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
                print(f'\nSTAGE: L31 timing on {name} cohort',flush=True)
                try:subprocess.run(command,check=True)
                except subprocess.CalledProcessError as error:
                    status.update(state='failed',returncode=error.returncode)
                    (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n');raise
                report(args.output,sources)
        (args.output/'status.json').write_text(json.dumps(dict(stage='Both cohorts finished',state='finished'),indent=2)+'\n')


if __name__=='__main__':main()
