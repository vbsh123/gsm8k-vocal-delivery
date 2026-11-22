"""Resume shared-GSM happy steering, then collect angry/happy on shared-three IDs in the other tasks."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

from diffa_gsm_shared_emotions import split_ids
from jenny_bundle_registry import Registry, sha

DATASETS = ('gsm8k', 'svamp', 'multiarith', 'bbh3', 'bbh5', 'ifeval')
PROTOCOL = 'diffa-shared-three-task-v1'


def prepare_task(name, template, registry, folder):
    import diffa_happy_nightly as task
    from diffa_collection_report import collection_bank
    neutral, shared, receipts, bank = {}, {}, {}, {}
    preferred = {r['question_id']: r['audio_sha256'] for r in template['target']['records']}
    preferred.update({p['question_id']: p['neutral']['audio_sha256'] for p in template['pairs']})
    for q in registry.neutral_ids(name):
        item = registry.choose(name, q, 'neutral', expected=preferred.get(q))
        if item is None:
            raise ValueError(f'Original neutral recording missing: {name}/{q}')
        neutral[q] = item['record']
        old = template['target']['bank'].get(q, template['bank'].get(q, {}))
        bank[q] = dict(old, question_id=q, question=item['question'], answer=item['answer'])
        emotional = {e: registry.choose(name, q, e, question=item['question'], answer=item['answer'], agreed=True)
                     for e in ('angry', 'happy', 'sad')}
        if all(emotional.values()):
            shared[q] = {e: entry['record'] for e, entry in emotional.items()}
            receipts[q] = {e: registry.labels[entry['record']['audio_sha256']] for e, entry in emotional.items()}
    train, test = split_ids(neutral, shared)
    bank = collection_bank(dict(template, bank=bank, collection_ids=list(bank)))
    target = dict(deepcopy(template['target']), records=[neutral[q] for q in test],
                  bank={q: bank[q] for q in test}, extraction_ids=train)
    split = dict(protocol=PROTOCOL, task=name, neutral_pool_ids=sorted(neutral), eligible_shared_ids=sorted(shared),
                 collection_ids=train, test_ids=test, seed=42, prompt=target['prompt'],
                 rule='Same question/text/reference; angry/happy/sad each e2v-agreed. Neutral label ignored. '
                      'Collect min(shared,floor(neutral_pool/2)); evaluate the remainder without emotion/outcome filtering.')
    task.winners.freeze(folder/'split.json', split)
    for emotion in ('angry', 'happy'):
        plan = dict(deepcopy(template), protocol=PROTOCOL, emotion=emotion, bank=bank, target=deepcopy(target),
                    collection_ids=train, test_ids=test, prompt=target['prompt'], collection_prompt=target['prompt'],
                    recordings=[dict(neutral[q], condition='neutral') for q in test],
                    pairs=[dict(question_id=q, neutral=neutral[q], emotion=shared[q][emotion]) for q in train],
                    classifier_receipts=[dict(question_id=q, labels=receipts[q]) for q in train], selection=split['rule'],
                    note='Exploratory shared-three split. Both collection and evaluation use the target zero-shot prompt. '
                         'No test outcome/emotion filtering. Old source descriptors are retained only as provenance.')
        task.winners.freeze(folder/emotion/'run_config.json', plan)
    print(f'{name}: shared={len(shared)}, collection={len(train)}, held-out={len(test)}; identical IDs for both emotions.', flush=True)


def vector_ready(plan, folder):
    import winning_diffa_sweeps as winners
    path = folder/'style-vectors.pt'
    if not path.exists():
        return False
    _, info = winners.read_vector(path, plan['emotion'])
    if (info['extraction_ids'] != plan['collection_ids'] or info['pairs'] != len(plan['collection_ids'])
            or info['collection_prompt'] != plan['collection_prompt'] or info['collection_tokens'] != plan['max_new_tokens']):
        raise ValueError(f'Existing vector does not match frozen collection: {path}')
    return True


def checked_answers(job, folder):
    from steer import read_rows
    target = job['target']
    records = {(r['question_id'],r['voice']):r for r in target['records']}
    done = set()
    for row in read_rows(folder/'results.jsonl', repair=True):
        key = row['question_id'],row['voice']
        strength = row.get('strength')
        if key not in records or strength not in (0.,1.) or (*key,strength) in done:
            raise ValueError('Unknown or duplicate shared-vector steering result.')
        if (row['audio_sha256'] != records[key]['audio_sha256'] or row['prompt'] != target['prompt']
                or row['seed'] != 42 or row['max_new_tokens'] != target['max_new_tokens']
                or row.get('decoder') != 'fast_dllm'):
            raise ValueError('Shared-vector saved audio/prompt/decoder recipe differs.')
        if strength and row.get('vector_sha256') != job['source']['vector_sha256']:
            raise ValueError('Shared-vector saved answer used another vector.')
        if not strength and row.get('interventions'):
            raise ValueError('Shared-vector neutral baseline must be unsteered.')
        done.add((*key,strength))
    return done


def report(root, gsm, names, tokenizer=None):
    import diffa_happy_nightly as task
    for name in names:
        folder = gsm if name == 'gsm8k' else root/name
        for emotion in ('angry', 'happy'):
            output = folder/emotion/'primary'
            print(f'\n{name}: shared-three {emotion} vector x+1', flush=True)
            if (output/'run_config.json').exists():
                task.report_steering(output, tokenizer)
            else:
                print('Not started.')
        compare_emotions(folder)


def compare_emotions(folder):
    import winning_diffa_sweeps as winners
    from rescore_audio_steering import paired
    from steer import read_rows
    groups = {}
    for emotion in ('angry', 'happy'):
        output = folder/emotion/'primary'
        if not (output/'run_config.json').exists():
            return
        job = json.loads((output/'run_config.json').read_text())
        scored, _ = winners.score_rows(job, read_rows(output/'results.jsonl'), output)
        groups[emotion] = {r['question_id']: r for r in scored if r.get('strength') == 1.}
    ids = sorted(set(groups['angry']) & set(groups['happy']))
    if not ids:
        return
    results = []
    for metric in groups['angry'][ids[0]]['metrics']:
        result = paired([(groups['angry'][q]['metrics'][metric], groups['happy'][q]['metrics'][metric]) for q in ids])
        results.append(dict(metric=metric, comparison='happy steering vs angry steering', **result))
        print(f'Happy vs angry steering / {metric}: N={len(ids)}; gain={result["gain_pp"]:+.2f} pp; p={result["mcnemar_exact_p"]:.4g}', flush=True)
    winners.atomic_text(folder/'happy-vs-angry-steering.json', json.dumps(dict(question_ids=ids, paired=results), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'run', 'summary'], default='run')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--audio-root', type=Path, default=Path('.'))
    parser.add_argument('--source-root', type=Path, default=Path('outputs/diffa-happy-winners'))
    parser.add_argument('--output', type=Path, default=Path('outputs/diffa-shared-three-tasks'))
    parser.add_argument('--gsm-output', type=Path, default=Path('outputs/diffa-gsm-shared-three-zeroshot-hashes'))
    args = parser.parse_args()
    if args.mode == 'summary':
        report(args.output, args.gsm_output, args.datasets)
        return
    import diffa_happy_nightly as task
    import diffa_angry_agreed as angry
    from diffa_bbh_angry125 import fill_neutral
    from steer import read_rows
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'queue.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = None
        jobs = []
        for name in args.datasets:
            folder = args.gsm_output if name == 'gsm8k' else args.output/name
            if name == 'gsm8k':
                for emotion in ('angry', 'happy'):
                    config = folder/emotion/'run_config.json'
                    if not config.exists() or not (folder/emotion/'style-vectors.pt').exists():
                        raise FileNotFoundError('Restore the COMPLETED shared-GSM angry/happy vectors and configs first; this queue does not recollect GSM.')
            elif not (folder/'split.json').exists():
                registry = registry or Registry(args.audio_root)
                template = json.loads((args.source_root/name/'run_config.json').read_text())
                prepare_task(name, template, registry, folder)
            plans = {e: json.loads((folder/e/'run_config.json').read_text()) for e in ('angry', 'happy')}
            if (plans['angry']['collection_ids'] != plans['happy']['collection_ids']
                    or plans['angry']['test_ids'] != plans['happy']['test_ids']
                    or set(plans['angry']['collection_ids']) & set(plans['angry']['test_ids'])):
                raise ValueError('Mismatched or overlapping emotion cohorts.')
            for emotion, plan in plans.items():
                task.winners.score_rows(dict(target=plan['target']), [], folder/emotion)
                jobs.append((name, emotion, plan, folder/emotion))
        if args.mode == 'prepare':
            return
        replay_path = Path('outputs/new-machine-neutral-replay/results.json')
        replay = json.loads(replay_path.read_text()) if replay_path.exists() else []
        if len(replay) < 10 or not all(row.get('identical_response') for row in replay):
            raise RuntimeError('Run bash run_diffa_replay.sh --run on this machine first. Need ten matching saved neutral responses before baseline reuse.')
        if shutil.disk_usage(args.output).free < 600*2**20:
            raise RuntimeError('Need 600 MiB free for collection checkpoints.')
        with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
            print('Waiting for shared DIFFA GPU queue.', flush=True)
            fcntl.flock(gpu, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                time.sleep(30)
            model = tokenizer = features = None
            previous = {}
            for name, emotion, plan, folder in jobs:
                ready = vector_ready(plan, folder)
                destination = folder/'primary'
                if ready:
                    source = task.descriptor(plan, folder) if emotion == 'happy' else angry.descriptor(plan, folder)
                    job = angry.job_for(source, plan['target'], [0., 1.])
                    destination.mkdir(exist_ok=True)
                    task.winners.freeze(destination/'run_config.json', job)
                    done = checked_answers(job, destination)
                    expected = {(r['question_id'],r['voice'],a) for r in plan['target']['records'] for a in (0.,1.)}
                    if expected <= done:
                        print(f'{name}/{emotion}: vector and all answers already complete; no inference.', flush=True)
                        task.report_steering(destination, tokenizer)
                        previous[name] = destination
                        continue
                if model is None:
                    from extract_vectors import load_model
                    model, tokenizer, features = load_model()
                if not ready:
                    task.collect(plan, folder, model, tokenizer, features)
                source = task.descriptor(plan, folder) if emotion == 'happy' else angry.descriptor(plan, folder)
                job = angry.job_for(source, plan['target'], [0., 1.])
                destination.mkdir(exist_ok=True)
                task.winners.freeze(destination/'run_config.json', job)
                task.winners.reuse_answers(job, destination)
                if name in previous:
                    task.reuse_neutral(previous[name], destination)
                checked_answers(job, destination)
                fill_neutral(plan, job, destination, model, tokenizer, features)
                task.winners.generate_job(job, destination, model, tokenizer, features)
                task.report_steering(destination, tokenizer)
                previous[name] = destination
                if emotion == 'happy':
                    compare_emotions(folder.parent)
        report(args.output, args.gsm_output, args.datasets, tokenizer)


if __name__ == '__main__':
    main()
