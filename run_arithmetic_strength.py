"""Reuse the arithmetic bare-task block vector and exact25-question holdout at another strength."""
import argparse
from copy import deepcopy
import fcntl
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import time

import run_arithmetic_steering as base
from steer import read_rows

follow = base.follow
winners = base.winners
PROTOCOL = 'diffa-bbh-arithmetic-existing-block-strength-v1'


def ready(source):
    config = source/'L31-answer-a2/run_config.json'
    if not config.exists() or not (source/'collection/style-vectors.pt').exists(): return False
    old = json.loads(config.read_text())
    baseline = [r for r in read_rows(config.parent/'results.jsonl') if r.get('strength')==0.]
    return len(baseline)==len(old['target']['records'])


def prepare(source, output, strength=3.):
    if not math.isfinite(strength) or strength<=0. or strength==2.:
        raise ValueError('Choose a finite positive follow-up strength other than the existing x2.')
    experiment = json.loads((source/'run_config.json').read_text())
    collection = json.loads((source/'collection/run_config.json').read_text())
    original = source/'L31-answer-a2/run_config.json'
    old = json.loads(original.read_text())
    state = base.night.load_state(collection, source/'collection')
    if (experiment['protocol']!=base.PROTOCOL or old['protocol']!=base.PROTOCOL
            or old['emotion']!='angry' or old['strengths']!=[0.,2.]
            or old['layer']!=31 or old['group']!='answer'
            or old['target']['task']!=base.pilot.TASK or old['target']['prompt']!=base.pilot.BARE_PROMPT
            or collection['prompt']!=base.pilot.BARE_PROMPT
            or state['completed_ids']!=experiment['collection_ids']
            or collection['collection_ids']!=experiment['collection_ids']
            or len(state['completed_ids'])!=25 or len(experiment['evaluation_ids'])!=25
            or set(experiment['evaluation_ids']) & set(experiment['collection_ids'])
            or [r['question_id'] for r in old['target']['records']]!=experiment['evaluation_ids']):
        raise ValueError('Need the completed original bare arithmetic block collection and unchanged holdout.')
    vector = source/'collection/style-vectors.pt'
    if winners.digest(vector)!=old['source']['vector_sha256']:
        raise ValueError('Original arithmetic block vector changed.')
    _, info = winners.read_vector(vector, 'angry')
    if info['extraction_ids']!=experiment['collection_ids'] or info['pairs']!=25:
        raise ValueError('Vector fitting IDs differ from the original disjoint split.')
    old_rows = follow.validate(old, original.parent)
    runtime = json.loads((source/'runtime.json').read_text())
    job = deepcopy(old)
    job.update(protocol=PROTOCOL, strengths=[0.,strength],
        parent_config_sha256=winners.digest(original), models=runtime['model_revisions'], packages=runtime['packages'])
    for part in ('source','target'):
        job[part]['vector']=str(vector.resolve())
    winners.freeze(output/'run_config.json', job)
    done = follow.validate(job, output, complete_baseline=False, repair=True)
    with (output/'results.jsonl').open('a') as stream:
        for row in old_rows.values():
            if row['strength']!=0.: continue
            key=row['question_id'],row['voice'],row['condition']
            if key not in done:
                stream.write(json.dumps(row)+'\n'); done[key]=row
    follow.validate(job, output)
    print(f'Arithmetic L31 answer x{strength:+g}: same25 held-out questions; existing25-pair '
          'bare angry-minus-neutral current-block vector. No recollection.', flush=True)
    return job


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('run','prepare','summary'),default='run')
    parser.add_argument('--strength',type=float,default=3.)
    parser.add_argument('--source',type=Path,default=Path('outputs/diffa-bbh-arithmetic-steering'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not math.isfinite(args.strength) or args.strength<=0. or args.strength==2.:
        parser.error('--strength must be finite, positive and different from the existing x2.')
    output=args.output or args.source/f'L31-answer-a{args.strength:g}'
    if output.resolve()==(args.source/'L31-answer-a2').resolve():
        parser.error('Use a separate output folder from the original x2 run.')
    if args.mode=='summary':
        if not (output/'run_config.json').exists():
            print('Arithmetic follow-up not prepared yet; it may be waiting for the original block collection.',flush=True);return
        job=json.loads((output/'run_config.json').read_text())
        if job['strengths']!=[0.,args.strength] or job['protocol']!=PROTOCOL:
            raise ValueError('Summary folder belongs to a different strength or experiment.')
        follow.validate(job,output)
        base.night.vanilla.task.report_steering(output)
        return
    output.mkdir(parents=True,exist_ok=True)
    with (output/'runner.lock').open('a') as own:
        try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('This arithmetic strength runner is already active.')
        if not ready(args.source):
            print('Waiting for the original arithmetic block vector and held-out neutral baseline.',flush=True)
            if args.mode=='prepare':return
            while not ready(args.source):time.sleep(30)
        job=prepare(args.source,output,args.strength)
        if args.mode=='prepare':return
        done=follow.validate(job,output)
        if len(done)==2*len(job['target']['records']):
            base.night.vanilla.task.report_steering(output);return
        if base.model_revisions()!=job['models'] or {k:version(k) for k in ('torch','transformers')}!=job['packages']:
            raise ValueError('Models or packages differ from the original arithmetic collection.')
        repo=Path(__file__).resolve().parent
        paths,_=follow.resolve_audio(job,[repo/'outputs/jenny-zip-audio',repo])
        with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
            print('Waiting for shared GPU lock and idle GPU; reusing the existing block vector.',flush=True)
            fcntl.flock(gpu,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                time.sleep(30)
            backend=base.night.DiffaBackend()
            original=next(r for r in done.values() if r['strength']==0.)
            record=dict(original,audio_file=paths[original['audio_sha256']])
            replay=backend.generate(record,dict(job['target'],recipe=base.night.vanilla.RECIPE),capture=False)
            if replay['response']!=original['response']:
                raise ValueError('Unsteered replay differs; no new strength responses generated.')
            winners.generate_job(follow.with_audio_paths(job,paths),output,backend.model,backend.tokenizer,backend.features)
            follow.validate(job,output)
            base.night.vanilla.task.report_steering(output,backend.tokenizer)


if __name__=='__main__':main()
