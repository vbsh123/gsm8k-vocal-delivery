"""Test four >=2pp candidates plus periodic3 on third-batch neutrals, then periodic3 on the old 460."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from evaluate_audio_steering import report
from steer_azure_holdout import select_recordings


def condition(schedule='always'):
    name='L12-acoustic-a1'+('' if schedule=='always' else '-'+schedule)
    value=dict(name=name,interventions=[dict(layer=12,group='acoustic',strength=1)])
    if schedule!='always':
        value['schedule']=schedule
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--path',type=Path,default=Path('outputs/azure-jenny-remaining-test'))
    p.add_argument('--num-examples',type=int,default=0,help='0 means all eligible third-batch neutrals.')
    p.add_argument('--output',type=Path,default=Path('outputs/steer-jenny-third-promoted'))
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    label='Same third-batch neutral recordings; candidates fixed from the second batch; no outcome/classifier filtering.'
    if args.summary_only:
        report(args.output,['original','angry_correct','forward20'],label)
        report(args.output/'second-set-periodic3',['original'],'Every-three-forwards diagnostic on the original second-set 460 recordings.')
        return
    import soundfile as sf
    records=select_recordings(args.path,args.num_examples,sf)
    args.output.mkdir(parents=True,exist_ok=True)
    cohort=args.output/'cohort'
    cohort.mkdir(exist_ok=True)
    path=cohort/'run_config.json'
    value=dict(recordings=records,selection=label)
    if path.exists() and json.loads(path.read_text())!=value:
        p.error('Frozen third-batch cohort changed; use another output folder')
    path.write_text(json.dumps(value,indent=2)+'\n')
    timing=Path('outputs/jenny-large/timing/original_mean')
    sources={
        'original':timing,
        'angry_correct':Path('outputs/steer-jenny-outcome-filters/matched/angry_correct_mean'),
        'forward20':Path('outputs/jenny-large/temporal-evaluation/forward_windows'),
    }
    commands=[]
    for method,source in sources.items():
        previous=json.loads((source/'run_config.json').read_text())
        vector=Path(previous['settings']['vectors'])
        if not vector.is_file():
            p.error(f'Missing fitted vector from {method}: {vector}')
        schedules=['always','periodic2','periodic3'] if method=='original' else ['always']
        conditions=[dict(name='baseline',interventions=[])]+[condition(s) for s in schedules]
        conditions_file=args.output/f'{method}-conditions.json'
        conditions_file.write_text(json.dumps(conditions,indent=2)+'\n')
        baseline=Path('outputs/steer-jenny-remaining200-L12') if method=='original' else args.output/'original'
        command=[sys.executable,'-u','evaluate_audio_steering.py','--source-run',str(cohort),
                 '--vectors',str(vector),'--methods',method,'--conditions-file',str(conditions_file),
                 '--num-examples',str(len(records)),'--output',str(args.output),
                 '--baseline-source',str(baseline),'--cohort-label',label]
        commands.append(command)
    original=json.loads((timing/'run_config.json').read_text())
    file=args.output/'second-set-periodic3-conditions.json'
    file.write_text(json.dumps([dict(name='baseline',interventions=[]),condition('periodic3')],indent=2)+'\n')
    commands.append([sys.executable,'-u','evaluate_audio_steering.py','--source-run',str(timing),
                     '--vectors',original['settings']['vectors'],'--methods','original','--conditions-file',str(file),
                     '--num-examples',str(len(original['recordings'])),'--output',str(args.output/'second-set-periodic3'),
                     '--baseline-source',str(timing),'--cohort-label','Every-three-forwards diagnostic on the original second-set 460 recordings.'])
    print(f'{len(records)} third-batch questions x 6 conditions including one shared baseline, plus 460 periodic3 answers on the second batch.',flush=True)
    for command in commands:
        subprocess.run(command+['--prepare-only'],check=True)
    if args.prepare_only:
        return
    with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
        print('Waiting for the shared GPU lock.',flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            print('Waiting for active GPU work.',flush=True)
            time.sleep(30)
        for command in commands:
            subprocess.run(command,check=True)
    report(args.output,['original','angry_correct','forward20'],label)
    report(args.output/'second-set-periodic3',['original'],'Every-three-forwards diagnostic on the original second-set 460 recordings.')


if __name__=='__main__':
    main()
