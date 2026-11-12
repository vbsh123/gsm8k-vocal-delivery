"""Wait for the current queue, collect L31 answer window vectors, and evaluate both cohorts."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from evaluate_audio_steering import copy_baselines, valid_baseline
from finish_jenny_finalists import FINALISTS
from run_jenny_late_layers import conditions
from steer import read_rows, summarize
from steer_audio import select_conditions


FINAL_LABEL='Final Azure cohort, all eligible neutral recordings; no classifier or outcome filtering.'


def pooled_report(output, late_root):
    result={}
    for method, folders, names in [
        ('Original mean L31', [late_root/'original_mean', late_root/'third-cohort/original_mean'], FINALISTS),
        ('L31 20-forward-window vectors', [output/'evaluation/second/forward20', output/'evaluation/third/forward20'], ['L31-answer-a1'])]:
        answers, seen, planned, selected = [], set(), 0, None
        for folder in folders:
            path=folder/'run_config.json'
            if not path.exists():
                continue
            config=json.loads(path.read_text())
            questions={r['question_id'] for r in config['recordings']}
            if questions & seen:
                raise ValueError('Cannot pool overlapping question cohorts')
            seen.update(questions)
            planned+=len(config['recordings'])
            selected=select_conditions(config['settings']['conditions'],names)
            answers.extend(read_rows(folder/'results.jsonl'))
        if selected:
            print(f'\nCOMBINED: {method}; {planned} planned recordings. Each row uses completed pairs only.',flush=True)
            result[method]=summarize(answers,selected)
    return result


def reuse_third_baselines(destination, source):
    config=json.loads((destination/'run_config.json').read_text())
    expected={r['clip_id']:r for r in config['recordings']}
    previous=json.loads((source/'run_config.json').read_text())
    previous={r['clip_id']:r for r in previous['recordings']}
    shared={r['clip_id']:r for r in read_rows(source/'results.jsonl')
            if r['clip_id'] in expected and previous.get(r['clip_id'])==expected[r['clip_id']]
            and valid_baseline(r,expected[r['clip_id']])}
    copy_baselines(destination/'results.jsonl',shared,expected)
    print(f'Reused {len(shared)} compatible final-cohort baselines.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--late-root',type=Path,default=Path('outputs/steer-jenny-late-layers'))
    p.add_argument('--collection',type=Path,default=Path('outputs/jenny-svd-probe'))
    p.add_argument('--third-source-run',type=Path,default=Path('outputs/steer-jenny-third-promoted/cohort'))
    p.add_argument('--third-baseline-source',type=Path,default=Path('outputs/steer-jenny-third-promoted/original'))
    p.add_argument('--output',type=Path,default=Path('outputs/jenny-L31-forward20'))
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    if args.summary_only:
        status=args.output/'status.json'
        if status.exists():
            print(status.read_text(),flush=True)
        pooled_report(args.output,args.late_root)
        return
    original=args.late_root/'original_mean'
    original_config=json.loads((original/'run_config.json').read_text())
    original_vector=Path(original_config['settings']['vectors'])
    source=json.loads((args.collection/'collection-config.json').read_text())
    third=json.loads((args.third_source_run/'run_config.json').read_text())
    train={r['question_id'] for r in source['pairs']}
    second_ids={r['question_id'] for r in original_config['recordings']}
    third_ids={r['question_id'] for r in third['recordings']}
    if train & (second_ids|third_ids) or second_ids & third_ids:
        p.error('Extraction/evaluation questions overlap')
    for record in list(source['manifest'].values())+original_config['recordings']+third['recordings']:
        if not Path(record['audio_path']).is_file():
            p.error(f'Missing recording: {record["audio_path"]}')
    args.output.mkdir(parents=True,exist_ok=True)
    vectors=args.output/'vectors'
    checkpoint=vectors/'forward_windows/style-vectors.pt'
    temporal_conditions=args.output/'forward20-conditions.json'
    temporal_conditions.write_text(json.dumps([
        dict(name='baseline',interventions=[]),
        dict(name='L31-answer-a1',interventions=[dict(layer=31,group='answer',strength=1)])],indent=2)+'\n')
    original_conditions=args.output/'original-conditions.json'
    original_conditions.write_text(json.dumps(conditions(),indent=2)+'\n')
    collect=[sys.executable,'-u','collect_temporal_vectors.py','--collection',str(args.collection),
             '--original-vector',str(original_vector),'--output',str(vectors),
             '--targets','31:answer','--modes','forward_windows']
    original_third=[sys.executable,'-u','evaluate_audio_steering.py',
                    '--source-run',str(args.third_source_run),'--vectors',str(original_vector),
                    '--methods','original_mean','--conditions-file',str(original_conditions),
                    '--num-examples',str(len(third['recordings'])),'--output',str(args.late_root/'third-cohort'),
                    '--baseline-source',str(args.third_baseline_source),'--cohort-label',FINAL_LABEL]
    evaluations=[]
    for name,source_run,baseline,records in [
        ('second',original,original,original_config['recordings']),
        ('third',args.third_source_run,args.third_baseline_source,third['recordings'])]:
        evaluations.append([sys.executable,'-u','evaluate_audio_steering.py',
                            '--source-run',str(source_run),'--vectors',str(checkpoint),
                            '--methods','forward20','--conditions-file',str(temporal_conditions),
                            '--num-examples',str(len(records)),'--output',str(args.output/'evaluation'/name),
                            '--baseline-source',str(baseline),
                            '--cohort-label',f'L31 answer 20-forward vectors learned from first-cohort pairs; {name} evaluation cohort; no outcome or classifier filter.'])
    print(f'Collect L31 answer on {len(source["pairs"])} training pairs (up to {2*len(source["pairs"])} generations).',flush=True)
    print(f'Test L31 forward20 x1 always-on on {len(second_ids)+len(third_ids)} recordings; finish two original L31 conditions on {len(third_ids)} final-cohort recordings.',flush=True)
    print('Preparing collection/configs on CPU only. Actual collection waits for the current finalist queue.',flush=True)
    subprocess.run(collect+['--prepare-only'],check=True)
    subprocess.run(original_third+['--prepare-only'],check=True)
    if args.prepare_only:
        return
    with (args.output/'queue.lock').open('w') as own_lock:
        try:
            fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('This L31 pipeline is already queued or running')
        def stage(name,command):
            (args.output/'status.json').write_text(json.dumps(dict(stage=name,state='running',time=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
            print(f'\nSTAGE: {name}',flush=True)
            try:
                subprocess.run(command,check=True)
            except subprocess.CalledProcessError as error:
                (args.output/'status.json').write_text(json.dumps(dict(stage=name,state='failed',returncode=error.returncode,time=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
                raise
        stage('Waiting for current steering/finalist queue',
              [sys.executable,'-u','wait_for_experiments.py','--kind','finalist-queue'])
        stage('Waiting for emotion/accuracy queue',
              [sys.executable,'-u','wait_for_experiments.py','--kind','azure-emotions'])
        with Path('outputs/jenny-task-vectors.lock').open('w') as gpu_lock:
            print('Waiting for the shared GPU lock.',flush=True)
            fcntl.flock(gpu_lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work.',flush=True)
                time.sleep(30)
            stage('Collect and fit L31 answer 20-forward vectors',collect)
            for name,command in zip(('Evaluate L31 forward20 on second cohort','Evaluate L31 forward20 on final cohort'),evaluations):
                stage(name,command)
            destination=args.late_root/'third-cohort/original_mean'
            reuse_third_baselines(destination,args.third_baseline_source)
            stage('Finish original mean L31 finalists on final cohort',
                  [sys.executable,'-u','steer_audio.py','--config',str(destination/'run_config.json'),
                   '--output',str(destination),'--conditions',*FINALISTS])
        summary=pooled_report(args.output,args.late_root)
        (args.output/'combined-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (args.output/'status.json').write_text(json.dumps(dict(stage='All requested stages finished',state='finished',time=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')


if __name__=='__main__':
    main()
