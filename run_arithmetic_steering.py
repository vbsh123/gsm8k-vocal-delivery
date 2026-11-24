"""Fit bare angry-minus-neutral current-block arithmetic vectors, then steer L31 x2 on disjoint questions."""
import argparse
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
import time

import torch
import run_arithmetic_audio_pilot as pilot
import emotion_block_followups as follow
from diffa_model_receipts import model_revisions

night = follow.night
winners = follow.winners
PROTOCOL = 'diffa-bbh-arithmetic-bare-block-fit25-eval25-x2-v1'
FIT_COUNT = 25


def parent_data(parent):
    config = parent/'run_config.json'
    plan = json.loads(config.read_text())
    ids = plan['question_order']
    if (plan['protocol'] != pilot.PROTOCOL or plan['prompts'] != pilot.PROMPTS
            or plan['source_git_blob'] != pilot.SOURCE_BLOB or plan['recipe'] != night.vanilla.RECIPE
            or len(ids) != 50 or len(set(ids)) != 50 or set(ids) != set(plan['bank'])
            or len(plan['records']) != 200
            or {(r['question_id'],r['condition']) for r in plan['records']} != {(q,c) for q in ids for c in pilot.CONDITIONS}):
        raise ValueError('Need the unchanged 50-question four-condition BBH arithmetic pilot.')
    rows = pilot.base.raw_rows(parent, plan)
    if any((q,c) not in rows for q in ids for c in ('neutral','angry')):
        raise ValueError('Complete the pilot\'s 50 neutral-bare and 50 angry-bare answers first.')
    return plan, rows


def prepare(parent, output):
    old, rows = parent_data(parent)
    selected = set(random.Random(44).sample(sorted(old['question_order']), FIT_COUNT))
    fit_ids = [q for q in old['question_order'] if q in selected]
    test_ids = [q for q in old['question_order'] if q not in selected]
    records = {(r['question_id'],r['condition']):r for r in old['records']}
    experiment = dict(protocol=PROTOCOL, model='diffa', name=pilot.TASK,
        parent_config_sha256=winners.digest(parent/'run_config.json'),
        parent_runtime_sha256=winners.digest(parent/'runtime.json'),
        collection_ids=fit_ids, evaluation_ids=test_ids, prompt=pilot.BARE_PROMPT,
        recipe=old['recipe'], layer=31, group='answer', strength=2.,
        selection='Seed44 random25/50 fitting questions; remaining25 held out. No E2V or outcome filtering.',
        collection='Current32-token answer-block means; matched chronological forward prefixes; equal question weight.',
        steering='L31 answer x+2 on neutral audio, full answer-buffer intervention, bare prompt unchanged.')
    winners.freeze(output/'run_config.json', experiment)
    collection = dict(protocol=PROTOCOL, model='diffa', name=pilot.TASK, emotion='angry',
        prompt=pilot.BARE_PROMPT, collection_prompt=pilot.BARE_PROMPT,
        bank=old['bank'], collection_ids=fit_ids,
        pairs=[dict(question_id=q, neutral=records[q,'neutral'], emotion=records[q,'angry']) for q in fit_ids],
        averaging=experiment['collection'], parent_config_sha256=experiment['parent_config_sha256'], **old['recipe'])
    winners.freeze(output/'collection/run_config.json', collection)
    print(f'Arithmetic: collect {len(fit_ids)} bare angry-minus-neutral pairs; '
          f'steer {len(test_ids)} disjoint neutral questions, L31 answer x+2.', flush=True)
    print('Prompt:\n'+pilot.BARE_PROMPT+'\nSelection: '+experiment['selection'], flush=True)
    return experiment, collection, old, rows


def collect(collection, folder, old_rows, backend):
    state = night.load_state(collection, folder)
    scoped = dict(collection, recipe=night.vanilla.RECIPE)
    for pair in collection['pairs'][len(state['completed_ids']):]:
        q = pair['question_id']
        print(f'Collect [{len(state["completed_ids"])+1}/{len(collection["pairs"])}] {q}: bare neutral + angry.', flush=True)
        neutral = backend.generate(pair['neutral'], scoped, capture=True)
        angry = backend.generate(pair['emotion'], scoped, capture=True)
        if neutral['response'] != old_rows[q,'neutral']['response'] or angry['response'] != old_rows[q,'angry']['response']:
            raise ValueError(f'{q}: captured generation differs from its saved bare response; pair not committed.')
        night.append_pair(collection, state, q, neutral, angry)
        night.vanilla.task.save_tensor(torch, state, folder/'collection-stream.pt')
        del neutral, angry
    night.finalize(collection, folder, state)


def steering_job(experiment, collection, old, old_rows, output):
    records = {(r['question_id'],r['condition']):r for r in old['records']}
    source = dict(name=pilot.TASK, task=pilot.TASK, emotion='angry', prompt=pilot.BARE_PROMPT,
        vector=str((output/'collection/style-vectors.pt').resolve()),
        bank=old['bank'], records=[records[q,'neutral'] for q in experiment['evaluation_ids']],
        extraction_ids=collection['collection_ids'], primary_metric='flexible-extract', **old['recipe'])
    source['vector_sha256'] = winners.digest(source['vector'])
    _, source['vector_info'] = winners.read_vector(source['vector'], 'angry')
    job = dict(protocol=PROTOCOL, emotion='angry', source=source, target=source,
        strengths=[0.,2.], layer=31, group='answer', mode='fixed', schedule='always')
    folder = output/'L31-answer-a2'
    winners.freeze(folder/'run_config.json', job)
    done = follow.validate(job, folder, complete_baseline=False, repair=True)
    with (folder/'results.jsonl').open('a') as stream:
        for q in experiment['evaluation_ids']:
            row = dict(old_rows[q,'neutral'], vector_sha256=None)
            key = q, row['voice'], row['condition']
            if key not in done:
                stream.write(json.dumps(row)+'\n'); done[key] = row
    follow.validate(job, folder)
    return folder, job


def report(output):
    path = output/'run_config.json'
    if not path.exists():
        print('Arithmetic collection/steering has not been prepared yet.', flush=True); return
    experiment = json.loads(path.read_text())
    collection = json.loads((output/'collection/run_config.json').read_text())
    state = night.load_state(collection, output/'collection')
    print(f'Arithmetic bare angry-minus-neutral: collected {len(state["completed_ids"])}/{FIT_COUNT}; '
          f'held-out N={len(experiment["evaluation_ids"])}.', flush=True)
    folder = output/'L31-answer-a2'
    if (folder/'run_config.json').exists():
        job = json.loads((folder/'run_config.json').read_text())
        follow.validate(job, folder)
        night.vanilla.task.report_steering(folder)
    else:
        print('L31 answer x+2 steering has not started yet.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run','prepare','summary'), default='run')
    parser.add_argument('--parent', type=Path, default=Path('outputs/diffa-bbh-arithmetic50'))
    parser.add_argument('--output', type=Path, default=Path('outputs/diffa-bbh-arithmetic-steering'))
    args = parser.parse_args()
    if args.mode == 'summary': report(args.output); return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'runner.lock').open('a') as own:
        try: fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Arithmetic steering runner already active; use --mode summary.')
        experiment, collection, old, rows = prepare(args.parent, args.output)
        pilot.scoring_preflight(args.output)
        if args.mode == 'prepare': return
        state = night.load_state(collection, args.output/'collection')
        if state['completed_ids'] == collection['collection_ids']:
            night.finalize(collection, args.output/'collection', state)
            folder, job = steering_job(experiment, collection, old, rows, args.output)
            if len(follow.validate(job, folder)) == 2*len(job['target']['records']):
                report(args.output); return
        records = [r for p in collection['pairs'] for r in (p['neutral'],p['emotion'])]
        records += [r for r in old['records'] if r['condition']=='neutral' and r['question_id'] in experiment['evaluation_ids']]
        for r in records:
            if winners.digest(r['audio_file']) != r['audio_sha256']:
                raise ValueError('Pilot recording missing or changed: '+r['audio_file'])
        original_runtime = json.loads((args.parent/'runtime.json').read_text())
        packages = {k:version(k) for k in ('torch','transformers')}
        if model_revisions() != original_runtime['model_revisions'] or packages != original_runtime['packages']:
            raise ValueError('Model revisions or packages differ from the original arithmetic pilot.')
        with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
            print('Waiting for shared GPU lock and idle GPU; no audio synthesis needed.', flush=True)
            fcntl.flock(gpu, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
                time.sleep(30)
            backend = night.DiffaBackend()
            winners.freeze(args.output/'runtime.json', dict(model_revisions=model_revisions(), packages=packages,
                runner_sha256=winners.digest(__file__), block_collector_sha256=winners.digest(night.blocks.__file__),
                streaming_collector_sha256=winners.digest(night.streaming.__file__)))
            q = experiment['evaluation_ids'][0]
            replay = backend.generate(rows[q,'neutral'], dict(old,prompt=pilot.BARE_PROMPT), capture=False)
            if replay['response'] != rows[q,'neutral']['response']:
                raise ValueError('Held-out neutral replay differs; no collection or steering started.')
            collect(collection, args.output/'collection', rows, backend)
            folder, job = steering_job(experiment, collection, old, rows, args.output)
            winners.generate_job(job, folder, backend.model, backend.tokenizer, backend.features)
            follow.validate(job, folder)
            night.vanilla.task.report_steering(folder, backend.tokenizer)


if __name__ == '__main__':
    main()
