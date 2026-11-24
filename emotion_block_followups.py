"""DIFFA-only strengths/transfer from completed overnight block vectors; primary-run archives."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path,PurePosixPath
import random
import stat
import subprocess
import time
import zipfile

import torch
import emotion_block_night as night
from diffa_layer_probe import resolve_audio,with_audio_paths
from diffa_model_receipts import MODELS,model_revisions
from steer import read_rows

winners=night.winners
ROLES=dict(old=('gsm8k','multiarith'),phi=('svamp','ifeval'),new=('bbh3','bbh5'))
DATASETS=tuple(n for group in ROLES.values() for n in group)
STRENGTHS=tuple(i/2 for i in range(-4,5))
PROTOCOL='emotion-block-night-followups-v1'
MARKER='EMOTION_NIGHT_ARCHIVE.json'


def primary_folders(root,role):
    folders=[root/'diffa'/name for name in ROLES[role]]
    if role=='phi':
        folders.append(root/'phi/gsm8k')
    return folders


def archive_night(root,role,destination,repo):
    """Only primary generations/collections/reports; no follow-up results, audio, dependencies."""
    root=root.resolve();repo=repo.resolve()
    folders=primary_folders(root,role)
    missing=[str(p) for p in folders if not (p/'run_config.json').is_file()]
    if missing:
        raise ValueError(f'Primary runs not prepared: {missing}')
    allowed={'.json','.jsonl','.csv','.pt'}
    excluded={'harness','upstream','.git','__pycache__','scoring','scoring-preflight','steering-comparison-scoring'}
    files=[]
    for folder in folders:
        for path in sorted(folder.rglob('*')):
            if path.is_file() and not path.is_symlink() and path.suffix in allowed and not excluded.intersection(path.relative_to(folder).parts):
                files.append(path)
    if not files:
        raise ValueError('No primary artifacts found.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        raise FileExistsError(f'Archive already exists; use another filename: {destination}')
    partial=destination.with_suffix('.zip.partial')
    manifest=dict(protocol=PROTOCOL,role=role,root=str(root.relative_to(repo)),files={},
        note='Primary overnight generations, agreed vectors, checkpoints, configs and scores only. Archive after generation finishes. No audio, model weights, venvs, or transfer/strength results.')
    try:
        with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for path in files:
                data=path.read_bytes()
                member=str(path.relative_to(repo))
                archive.writestr(member,data)
                manifest['files'][member]=hashlib.sha256(data).hexdigest()
            archive.writestr(MARKER,json.dumps(manifest,indent=2)+'\n')
        with zipfile.ZipFile(partial) as archive:
            if archive.testzip():
                raise ValueError('Primary archive failed CRC verification.')
        partial.rename(destination)
    finally:
        partial.unlink(missing_ok=True)
    destination.with_suffix('.zip.sha256').write_text(f'{winners.digest(destination)}  {destination.name}\n')
    print(f'DOWNLOAD: {destination.resolve()}\nSize: {destination.stat().st_size/2**20:.1f} MiB; {len(files)} primary artifact files.',flush=True)
    return destination


def restore(archives,repo,root):
    repo=repo.resolve();root=root.resolve()
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            receipt=json.loads(archive.read(MARKER))
            if receipt['protocol']!=PROTOCOL:
                raise ValueError('Wrong archive protocol.')
            for name,expected in receipt['files'].items():
                member=PurePosixPath(name)
                if member.is_absolute() or '..' in member.parts or stat.S_ISLNK(archive.getinfo(name).external_attr>>16):
                    raise ValueError('Unsafe archive member.')
                target=repo.joinpath(*member.parts)
                if not target.resolve().is_relative_to(root):
                    raise ValueError('Archive member is outside the primary overnight output directory.')
                data=archive.read(name)
                if hashlib.sha256(data).hexdigest()!=expected:
                    raise ValueError(f'Archive member hash changed: {name}')
                if target.exists():
                    if target.read_bytes()!=data:
                        raise ValueError(f'Existing primary artifact differs; left unchanged: {target}')
                    continue
                target.parent.mkdir(parents=True,exist_ok=True)
                tmp=target.with_name(target.name+'.restore-tmp')
                try:
                    with tmp.open('xb') as stream:stream.write(data)
                    tmp.rename(target)
                finally:
                    tmp.unlink(missing_ok=True)
        print(f'Imported exact primary artifacts: {path}',flush=True)


def complete_case(root,name,emotions=None):
    folder=root/'diffa'/name
    plan=json.loads((folder/'run_config.json').read_text())
    if plan['protocol']!=night.PROTOCOL or plan['model']!='diffa':
        raise ValueError('Need the new DIFFA overnight block-collection run, not an earlier vector.')
    rows=night.raw_rows(folder,plan)
    if len(rows)!=4*len(plan['question_order']):
        raise ValueError(f'{name}: finish all four natural deliveries first.')
    runtime=json.loads((folder/'runtime.json').read_text())
    result={}
    for e in night.EMOTIONS if emotions is None else emotions:
        p=json.loads((folder/e/'run_config.json').read_text())
        state=night.load_state(p,folder/e)
        if state['completed_ids']!=p['collection_ids']:
            raise ValueError(f'{name}/{e}: finish collection first.')
        expected=night.steering_job(folder,plan,p)
        own=folder/e/'L31-answer-a1'
        old=json.loads((own/'run_config.json').read_text())
        if (old['source']['vector_sha256']!=expected['source']['vector_sha256']
                or old['target']['prompt']!=plan['prompt']
                or old['target']['records']!=expected['target']['records']):
            raise ValueError('Overnight own-vector steering config differs.')
        from diffa_layer_probe import validate
        done=validate(expected,own)
        if len(done)!=2*len(expected['target']['records']):
            raise ValueError(f'{name}/{e}: finish overnight x+1 steering first.')
        vector=torch.load(expected['source']['vector'],map_location='cpu',weights_only=True)
        if not torch.equal(vector['pair_mean_sums'][e],state['difference_sum'].float()):
            raise ValueError('Final vector differs from its captured collection sum.')
        result[e]=dict(descriptor=expected['source'],rows=list(done.values()),models=runtime['model_revisions'],
                       collector_sha256=runtime['block_collector_sha256'])
    return result


def validate(job,folder,complete_baseline=True,repair=False):
    source,target=job['source'],job['target']
    if winners.digest(source['vector'])!=source['vector_sha256']:
        raise ValueError('Follow-up source vector changed.')
    _,info=winners.read_vector(source['vector'],job['emotion'])
    if set(info['extraction_ids']) & {r['question_id'] for r in target['records']}:
        raise ValueError('Source fitting questions overlap target evaluation.')
    expected={(r['question_id'],r['voice']):r for r in target['records']}
    done={}
    for row in read_rows(folder/'results.jsonl',repair=repair):
        key=row['question_id'],row['voice'],row['condition']
        strength=row.get('strength')
        interventions=[dict(layer=31,group='answer',strength=strength)] if strength else []
        record=expected.get(key[:2])
        if (record is None or key in done or strength not in job['strengths']
                or row['condition']!=winners.condition(strength)
                or row.get('interventions')!=interventions
                or row.get('vector_sha256')!=(source['vector_sha256'] if strength else None)
                or not winners.reusable(row,{},record,target)):
            raise ValueError('Follow-up response cache differs from the frozen recipe.')
        done[key]=row
    if complete_baseline and sum(r['strength']==0. for r in done.values())!=len(expected):
        raise ValueError('Follow-up neutral baseline incomplete.')
    return done


def prepare(root,output,role,kind):
    names=DATASETS if kind in ('transfer','both') else ROLES[role]
    cases={}
    for name in names:
        try:
            cases[name]=complete_case(root,name)
        except FileNotFoundError as error:
            raise ValueError(f'Missing completed {name} artifacts. For transfer, import the other machines primary-run archives first.') from error
    models=[entry['models'] for case in cases.values() for entry in case.values()]
    if set(models[0])!=set(MODELS) or any(m!=models[0] for m in models):
        raise ValueError('Dataset vectors were collected with different DIFFA model revision receipts.')
    collectors={entry['collector_sha256'] for case in cases.values() for entry in case.values()}
    if len(collectors)!=1:
        raise ValueError('Dataset vectors used different block-collector implementations.')
    jobs=[]
    for task in (('strengths','transfer') if kind=='both' else (kind,)):
        for emotion in night.EMOTIONS:
            for target_name in ROLES[role]:
                sources=[target_name] if task=='strengths' else [n for n in DATASETS if n!=target_name]
                for source_name in sources:
                    target=cases[target_name][emotion]['descriptor']
                    source=cases[source_name][emotion]['descriptor']
                    job=dict(protocol=PROTOCOL,kind=task,emotion=emotion,source=source,target=target,
                        strengths=list(STRENGTHS) if task=='strengths' else [0.,1.],
                        layer=31,group='answer',mode='fixed',schedule='always',models=models[0])
                    if set(source['extraction_ids']) & {r['question_id'] for r in target['records']}:
                        raise ValueError('Source collection overlaps target evaluation.')
                    label=target_name if task=='strengths' else f'{target_name}__from__{source_name}'
                    folder=output/task/emotion/label
                    winners.freeze(folder/'run_config.json',job)
                    done=validate(job,folder,complete_baseline=False,repair=True)
                    with (folder/'results.jsonl').open('a') as stream:
                        for row in cases[target_name][emotion]['rows']:
                            if row['strength']!=0. and task!='strengths':continue
                            if row['strength'] not in job['strengths']:continue
                            key=row['question_id'],row['voice'],row['condition']
                            if key not in done:
                                stream.write(json.dumps(row)+'\n');done[key]=row
                    done=validate(job,folder)
                    pending=len(target['records'])*len(job['strengths'])-len(done)
                    print(f'{task}/{emotion}: {source_name} -> {target_name}; {pending} generations pending; frozen target N={len(target["records"])}.',flush=True)
                    jobs.append((folder,job))
    manifest=dict(protocol=PROTOCOL,role=role,kind=kind,models=models[0],strengths=list(STRENGTHS),
        jobs=[dict(folder=str(folder.resolve()),config_sha256=winners.digest(folder/'run_config.json')) for folder,_ in jobs],
        note='Existing block vectors, exact target own holdout and recipe. No recollection, normalization, outcome selection, or Phi steering. Unadjusted exploratory paired tests.')
    winners.freeze(output/f'{role}-{kind}-manifest.json',manifest)
    return jobs


def summarize(output,role,kind):
    for task in (('strengths','transfer') if kind=='both' else (kind,)):
        for emotion in night.EMOTIONS:
            for folder in sorted((output/task/emotion).glob('*')):
                config=folder/'run_config.json'
                if config.exists() and json.loads(config.read_text())['target']['name'] in ROLES[role]:
                    job=json.loads(config.read_text())
                    validate(job,folder)
                    print(f'\n{task}/{emotion}',flush=True)
                    night.vanilla.task.report_steering(folder)


def import_available(root,role,repo,directory=Path('/workspace')):
    for other,names in ROLES.items():
        if other==role:
            continue
        if all((root/'diffa'/name/'run_config.json').exists() for name in names):
            continue
        archive=directory/f'emotion-night-{other}.zip'
        if archive.is_file():
            restore([archive],repo,root)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role',choices=tuple(ROLES))
    parser.add_argument('--mode',choices=('prepare','run','summary','archive-night','import'),default='run')
    parser.add_argument('--kind',choices=('strengths','transfer','both'),default='both')
    parser.add_argument('--root',type=Path,default=Path('outputs/emotion-block-night'))
    parser.add_argument('--output',type=Path,default=Path('outputs/emotion-block-followups'))
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--archives',nargs='+',type=Path)
    parser.add_argument('--audio-root',type=Path,action='append')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parent
    if args.mode=='archive-night':
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        archive_night(args.root,args.role,args.archive or Path('/workspace')/f'emotion-night-{args.role}-{stamp}.zip',repo)
        return
    if args.mode=='import':
        if not args.archives:parser.error('--mode import requires --archives')
        restore(args.archives,repo,args.root)
        return
    if args.mode=='summary':
        summarize(args.output,args.role,args.kind)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/f'{args.role}-runner.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.kind in ('transfer','both'):
            import_available(args.root,args.role,repo)
        jobs=prepare(args.root,args.output,args.role,args.kind)
        if args.mode=='prepare':return
        if all(len(validate(job,folder))==len(job['target']['records'])*len(job['strengths']) for folder,job in jobs):
            summarize(args.output,args.role,args.kind)
            return
        roots=[*(args.audio_root or [repo/'outputs/jenny-zip-audio']),repo]
        paths={}
        for _,job in jobs:
            found,_=resolve_audio(job,roots)
            paths.update(found)
        if model_revisions()!=jobs[0][1]['models']:
            raise ValueError('Active DIFFA model revision receipts differ from source vectors.')
        with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
            print('Waiting for existing GPU work to finish.',flush=True)
            fcntl.flock(gpu,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                time.sleep(30)
            from extract_vectors import load_model
            model,tokenizer,features=load_model()
            # One unchanged neutral replay per target before applying new interventions.
            checked=set()
            for folder,job in jobs:
                target=job['target']['name']
                if target not in checked:
                    original=next(r for r in validate(job,folder).values() if r['strength']==0.)
                    record=dict(original,audio_file=paths[original['audio_sha256']])
                    response,_=night.vanilla.task.generate(record,job['target'],job['target']['prompt'],model,tokenizer,features)
                    if response!=original['response']:
                        raise ValueError(f'{target}: neutral replay differs on this machine; no new steering started.')
                    checked.add(target)
            for folder,job in jobs:
                done=validate(job,folder)
                if len(done)==len(job['target']['records'])*len(job['strengths']):continue
                winners.generate_job(with_audio_paths(job,paths),folder,model,tokenizer,features)
                validate(job,folder)
                night.vanilla.task.report_steering(folder,tokenizer)


if __name__=='__main__':
    main()
