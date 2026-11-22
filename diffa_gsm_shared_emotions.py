"""GSM happy/angry vectors on identical three-emotion-agreed IDs, at most half the pool."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import random
import shutil
import subprocess
import time


def split_ids(neutral_ids, shared_ids):
    pool = set(neutral_ids)
    shared = sorted(set(shared_ids))
    if not set(shared) <= pool:
        raise ValueError('Shared collection IDs must have neutral audio.')
    shuffled = list(shared)
    random.Random(42).shuffle(shuffled)
    train = sorted(shuffled[:min(len(shared), len(pool)//2)])
    test = sorted(pool - set(train))
    if not train or not test:
        raise ValueError('Need nonempty collection and held-out cohorts.')
    return train, test


def prepare(args):
    import diffa_happy_nightly as task
    from check_sad_audio import rows
    from shared_emotion_counts import MODEL

    root = Path('outputs')
    source = json.loads((args.source / 'run_config.json').read_text())
    if source['name'] != 'gsm8k':
        raise ValueError('Expected original GSM happy configuration.')
    labels = {}
    for path in sorted(root.rglob('results-emotion2vec.jsonl')):
        for row in rows(path):
            if (row.get('model') == MODEL and row.get('processing_mode') == 'whole_recording'
                    and row.get('classification_status') == 'classified'):
                labels[row.get('audio_sha256') or row.get('clip_id')] = row
    candidates = {}
    for path in sorted(root.rglob('plan.jsonl')):
        for row in rows(path):
            q = row.get('question_id') or ''
            style = row.get('style')
            if (not q.startswith('gsm8k_test_') or row.get('provider') != 'azure'
                    or row.get('voice') != task.VOICE or style not in ('neutral', 'angry', 'happy', 'sad')
                    or not row.get('audio_path') or not (path.parent / row['audio_path']).is_file()):
                continue
            record = task.make_record(row, path.parent)
            candidates.setdefault(q, {}).setdefault(style, []).append((row, record))
    preferred = {r['question_id']: r['audio_sha256'] for r in source['target']['records']}
    preferred.update({p['question_id']: p['neutral']['audio_sha256'] for p in source['pairs']})
    neutral = {}
    triples = {}
    bank = {}
    receipts = {}
    for q, styles in sorted(candidates.items()):
        options = styles.get('neutral', [])
        if not options:
            continue
        if q in preferred:
            options = [item for item in options if item[1]['audio_sha256'] == preferred[q]]
            if not options:
                raise ValueError(f'Original neutral audio changed or missing: {q}')
        row, record = options[0]
        neutral[q] = record
        bank[q] = dict(question_id=q, question=row['question'], answer=record['reference_answer'])
        recognized = {}
        for emotion in ('angry', 'happy', 'sad'):
            for emotional_row, emotional_record in styles.get(emotion, []):
                label = labels.get(emotional_record['audio_sha256'], {})
                if (label.get('emotion') == emotion and emotional_row['question'] == row['question']
                        and emotional_record['reference_answer'] == record['reference_answer']):
                    recognized[emotion] = emotional_record
                    break
        if len(recognized) == 3:
            triples[q] = recognized
            receipts[q] = {e: labels[r['audio_sha256']] for e, r in recognized.items()}
    train, test = split_ids(neutral, triples)
    target = deepcopy(source['target'])
    target.update(records=[neutral[q] for q in test], bank={q: bank[q] for q in test}, extraction_ids=train)
    target['reuse_runs'] = list(dict.fromkeys([str(args.source / 'delivery'), *target.get('reuse_runs', [])]))
    split = dict(protocol='diffa-gsm-shared-three-emotions-v1', neutral_pool_ids=sorted(neutral),
                 eligible_shared_ids=sorted(triples), collection_ids=train, test_ids=test, seed=42,
                 rule='All three emotions must be recognized on the same text/reference. Neutral recognition ignored. '
                      'Collection=min(shared_count,floor(available_neutral_count/2)); remainder is test.',
                 source_config_sha256=task.winners.digest(args.source / 'run_config.json'),
                 original_collection_ids=source['collection_ids'], original_test_ids=source['test_ids'],
                 note='Exploratory new split; do not pool with original 425/718 evidence.')
    task.winners.freeze(args.output / 'split.json', split)
    plans = {}
    for emotion in ('happy', 'angry'):
        plan = deepcopy(source)
        plan.update(protocol=split['protocol'], emotion=emotion, target=deepcopy(target), bank=bank,
                    collection_ids=train, test_ids=test,
                    pairs=[dict(question_id=q, neutral=neutral[q], emotion=triples[q][emotion]) for q in train],
                    classifier_receipts=[dict(question_id=q, labels=receipts[q]) for q in train],
                    selection=split['rule'], parent_config_sha256=split['source_config_sha256'])
        folder = args.output / emotion
        folder.mkdir(parents=True, exist_ok=True)
        task.winners.freeze(folder / 'run_config.json', plan)
        plans[emotion] = plan
    print(f'GSM neutral pool={len(neutral)}, shared eligible={len(triples)}, half cap={len(neutral)//2}.', flush=True)
    print(f'Both vectors: SAME {len(train)} collection IDs; SAME {len(test)} held-out neutral IDs. L31 answer x+1.', flush=True)
    return plans


def report(output, tokenizer=None):
    import diffa_happy_nightly as task
    for emotion in ('happy', 'angry'):
        folder = output / emotion / 'primary'
        print(f'\nGSM shared collection — {emotion} vector', flush=True)
        if (folder / 'run_config.json').exists():
            task.report_steering(folder, tokenizer)
        else:
            print('Steering not started yet.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'run', 'summary'], default='run')
    parser.add_argument('--source', type=Path, default=Path('outputs/diffa-happy-winners/gsm8k'))
    parser.add_argument('--output', type=Path, default=Path('outputs/diffa-gsm-shared-three'))
    args = parser.parse_args()
    if args.mode == 'summary':
        report(args.output)
        return
    import diffa_happy_nightly as task
    import diffa_angry_agreed as angry
    from diffa_bbh_angry125 import fill_neutral
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'queue.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'split.json').exists():
            split = json.loads((args.output / 'split.json').read_text())
            if split['source_config_sha256'] != task.winners.digest(args.source / 'run_config.json'):
                raise ValueError('Original source changed.')
            plans = {e: json.loads((args.output / e / 'run_config.json').read_text()) for e in ('happy', 'angry')}
            print(f'Resuming frozen split: {len(split["collection_ids"])} collection / {len(split["test_ids"])} test.', flush=True)
        else:
            plans = prepare(args)
        for emotion, plan in plans.items():
            task.winners.score_rows(dict(target=plan['target']), [], args.output / emotion)
        if args.mode == 'prepare':
            return
        if shutil.disk_usage(args.output).free < 600 * 2**20:
            raise RuntimeError('Need 600 MiB free for running aggregate checkpoints.')
        with Path('outputs/jenny-task-vectors.lock').open('a') as lock:
            print('Waiting for shared DIFFA GPU queue and idle GPU.', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                time.sleep(30)
            from extract_vectors import load_model
            model, tokenizer, features = load_model()
            previous = None
            for emotion, plan in plans.items():
                folder = args.output / emotion
                task.collect(plan, folder, model, tokenizer, features)
                source = task.descriptor(plan, folder) if emotion == 'happy' else angry.descriptor(plan, folder)
                job = angry.job_for(source, plan['target'], [0., 1.])
                destination = folder / 'primary'
                destination.mkdir(exist_ok=True)
                task.winners.freeze(destination / 'run_config.json', job)
                task.winners.reuse_answers(job, destination)
                if previous is not None:
                    task.reuse_neutral(previous, destination)
                fill_neutral(plan, job, destination, model, tokenizer, features)
                task.winners.generate_job(job, destination, model, tokenizer, features)
                task.report_steering(destination, tokenizer)
                previous = destination
            report(args.output, tokenizer)


if __name__ == '__main__':
    main()
