"""Original-protocol sad: exact original happy/angry IDs, collection, delivery and L31 answer +1."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

import diffa_happy_nightly as task

PROTOCOL = 'diffa-original-sad-matched-cohorts-v1'


def prepare(name, source, folder, inventory):
    template = json.loads(source.read_text())
    if template['name'] != name or template['emotion'] != 'happy' or template['seed'] != 42:
        raise ValueError('Expected original happy frozen plan, not a shared-three or other emotion plan.')
    from export_diffa_cohorts import make_cohort
    make_cohort(template, template['bank'])  # Exact ordered IDs, disjointness and neutral hashes.
    plan = deepcopy(template)
    ids = template['collection_ids']+template['test_ids']
    neutral = {p['question_id']:p['neutral'] for p in template['pairs']}
    neutral.update({r['question_id']:r for r in template['recordings'] if r['condition']=='neutral'})
    sad = {}
    for q in ids:
        if task.winners.digest(neutral[q]['audio_file']) != neutral[q]['audio_sha256']:
            raise ValueError(f'Original neutral recording changed: {name}/{q}')
        bank = template['bank'][q]
        options = {}
        for row, parent in inventory.get((q,'sad'),[]):
            if row['question'] == bank['question'] and task.normal_answer(row.get('answer')) == bank['answer']:
                record = task.make_record(row,parent)
                options[record['audio_sha256']] = record
        if len(options) != 1:
            raise ValueError(f'{name}/{q}: need one byte-distinct existing sad recording matching text/reference; found {len(options)}. No TTS or question substitution.')
        sad[q] = next(iter(options.values()))
    plan.update(protocol=PROTOCOL,emotion='sad',
                original_happy_plan=str(source.absolute()),original_happy_plan_sha256=task.winners.digest(source),
                pairs=[dict(question_id=q,neutral=neutral[q],emotion=sad[q]) for q in template['collection_ids']],
                recordings=[dict((neutral if style=='neutral' else sad)[q],condition=style)
                            for q in template['test_ids'] for style in ('neutral','sad')],
                selection='Exact ordered original angry/happy collection and evaluation IDs. No e2v or response-accuracy filtering.',
                note='Sad-minus-neutral collection; original collection/evaluation prompts and budgets preserved. '
                     'Reuse compatible neutral answers; evaluate unsteered sad and L31 answer +1 steering.')
    # Original happy deliveries are also a valid source of the same neutral answers.
    plan['target']['reuse_runs'] = list(dict.fromkeys([*plan['target']['reuse_runs'],str((source.parent/'delivery').absolute())]))
    folder.mkdir(parents=True,exist_ok=True)
    task.winners.freeze(folder/'run_config.json',plan)
    print(f'{name}: exact original {len(plan["collection_ids"])}/{len(plan["test_ids"])} collection/evaluation IDs; existing sad audio, no label filtering.',flush=True)
    return plan


def steering_job(plan,folder):
    source = task.descriptor(plan,folder)
    return dict(protocol=PROTOCOL,kind='sad-steering',source=source,target=source,strengths=[0.,1.],
                layer=31,group='answer',schedule='always',mode='fixed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','run','summary'),default='run')
    parser.add_argument('--datasets',choices=task.DATASETS,nargs='+',default=list(task.DATASETS))
    parser.add_argument('--source-root',type=Path,default=Path('outputs/diffa-happy-winners'))
    parser.add_argument('--output',type=Path,default=Path('outputs/diffa-sad-winners'))
    args = parser.parse_args()
    if args.mode == 'summary':
        task.summarize(args.output,args.datasets)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'queue.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory = task.audio_inventory(Path('outputs'),styles=('sad',))
        plans = []
        for name in args.datasets:
            source, folder = args.source_root/name/'run_config.json', args.output/name
            plan = prepare(name,source,folder,inventory)
            delivery = folder/'delivery'
            delivery.mkdir(exist_ok=True)
            task.winners.freeze(delivery/'run_config.json',dict(protocol=PROTOCOL,target=plan['target'],conditions=['neutral','sad']))
            task.reuse_delivery(plan,delivery)
            from diffa_collection_report import collection_bank
            collection_bank(plan)
            task.winners.score_rows(dict(target=plan['target']),[],delivery)
            plans.append((plan,folder))
        task.winners.freeze(args.output/'queue_manifest.json',dict(protocol=PROTOCOL,datasets=args.datasets,
            emotion='sad',strength=1.,layer=31,group='answer',stages=['collection','delivery','primary'],
            cohort='Exact original angry/happy IDs and prompts; no classifier filtering; existing audio only.'))
        if args.mode == 'prepare':
            return
        if shutil.disk_usage(args.output).free < 600*2**20:
            raise RuntimeError('Need 600 MiB free for aggregate checkpoints and answers.')
        with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
            print('Waiting for original happy/shared DIFFA GPU lock; no model loaded while waiting.',flush=True)
            fcntl.flock(gpu,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                print('Waiting for active GPU work to finish.',flush=True)
                time.sleep(30)
            from extract_vectors import load_model
            model,tokenizer,features = load_model()
            task.winners.atomic_text(args.output/'runtime.json',json.dumps(dict(
                runner_sha256=task.winners.digest(__file__),collector_sha256=task.winners.digest(task.__file__),
                generation_sha256=task.winners.digest(Path(__file__).with_name('extract_vectors.py')),
                model_config=model.config.to_dict()),indent=2,default=str)+'\n')
            for plan,folder in plans:
                task.collect(plan,folder,model,tokenizer,features)
                task.report_collection(plan,folder,tokenizer)
                task.deliver(plan,folder/'delivery',model,tokenizer,features)
                task.report_delivery(plan,folder/'delivery',tokenizer)
                output = folder/'strength-1'
                output.mkdir(exist_ok=True)
                job = steering_job(plan,folder)
                task.winners.freeze(output/'run_config.json',job)
                task.reuse_neutral(folder/'delivery',output)
                task.winners.generate_job(job,output,model,tokenizer,features)
                task.report_steering(output,tokenizer)


if __name__ == '__main__':
    main()
