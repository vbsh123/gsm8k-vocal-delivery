"""Fit a classifier-agreed happy direction; keep the original 718 GSM test IDs."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

import diffa_happy_nightly as happy
from steer import read_rows


def agreed_audio(root):
    """Join each label to its exact audio plan, not just a question/style name."""
    found = {}
    for path in sorted(root.rglob('results-emotion2vec.jsonl')):
        if not (path.parent/'plan.jsonl').exists(): continue
        labels = {r['audio_path']: r for r in read_rows(path) if r.get('audio_path')}
        for row in read_rows(path.parent/'plan.jsonl'):
            if (row.get('provider') != 'azure' or row.get('voice') != happy.VOICE
                    or not row.get('question_id') or row.get('style') not in ('neutral', 'happy')): continue
            label = labels.get(row.get('audio_path'), {})
            if label.get('emotion') != row['style']: continue
            audio = path.parent/row['audio_path']
            if not audio.is_file(): continue
            record = happy.make_record(row, path.parent)
            key = row['question_id'], row['style']
            found.setdefault(key, []).append((row, record, dict(
                label_file=str(path.absolute()), label_sha256=happy.winners.digest(path),
                audio_sha256=record['audio_sha256'], predicted_emotion=label['emotion'])))
    return found


def prepare(source, output, pool, inventory):
    original = json.loads((source/'run_config.json').read_text())
    plan = deepcopy(original)
    if original['name'] != 'gsm8k': raise ValueError('This run is GSM-only.')
    ids = original['collection_ids']
    if pool == 'original500':
        import torch
        state = torch.load(original['target']['vector'], map_location='cpu', weights_only=True)
        ids = state['extraction_question_ids']
    if set(ids) & set(original['test_ids']): raise ValueError('Collection/test overlap.')
    existing = {p['question_id']: p for p in original['pairs']}
    pairs = []; receipts = []
    for qid in ids:
        records = {}; selected = {}
        for style, key in [('neutral', 'neutral'), ('happy', 'emotion')]:
            candidates = inventory.get((qid, style), [])
            if qid in existing:
                candidates = [item for item in candidates if item[1]['audio_sha256'] == existing[qid][key]['audio_sha256']]
            if not candidates: break
            row, record, receipt = candidates[0]
            records[key] = record; selected[style] = receipt
            if qid not in plan['bank']:
                plan['bank'][qid] = dict(question_id=qid, question=row['question'], answer=happy.normal_answer(row['answer']))
            if (row['question'] != plan['bank'][qid]['question'] or
                    record['reference_answer'] != happy.normal_answer(plan['bank'][qid]['answer'])):
                raise ValueError('Classifier audio differs from the frozen question/reference.')
        if len(records) != 2: continue
        pairs.append(dict(question_id=qid, **records)); receipts.append(dict(question_id=qid, **selected))
    if not pairs: raise ValueError('No neutral/happy-agreed collection pairs with audio available.')
    plan.update(protocol='diffa-gsm-happy-agreed-v1', collection_ids=[p['question_id'] for p in pairs], pairs=pairs,
                classifier='emotion2vec', classifier_receipts=receipts, selection_pool=pool,
                selection='Both neutral and happy must match emotion2vec; no correctness filtering.',
                parent_config_sha256=happy.winners.digest(source/'run_config.json'))
    output.mkdir(parents=True, exist_ok=True)
    happy.winners.freeze(output/'run_config.json', plan)
    print(f'{len(pairs)} agreed collection pairs; {len(plan["test_ids"])} unchanged held-out neutral questions.', flush=True)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['prepare', 'run', 'summary'], default='run')
    p.add_argument('--pool', choices=['existing425', 'original500'], required=True)
    p.add_argument('--source', type=Path, default=Path('outputs/diffa-happy-winners/gsm8k'))
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    output = args.output or Path('outputs/diffa-gsm-happy-agreed')/args.pool
    if args.mode == 'summary':
        if (output/'strength-1'/'run_config.json').exists(): happy.report_steering(output/'strength-1')
        else: print('Filtered-vector steering not prepared yet.')
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output/'queue.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = output/'run_config.json'
        if frozen.exists():
            plan = json.loads(frozen.read_text())
            if plan['selection_pool'] != args.pool or plan['parent_config_sha256'] != happy.winners.digest(args.source/'run_config.json'):
                raise ValueError('Filtered collection source changed.')
        else: plan = prepare(args.source, output, args.pool, agreed_audio(Path('outputs')))
        if args.mode == 'prepare': return
        destination = output/'strength-1'; destination.mkdir(exist_ok=True)
        happy.reuse_neutral(args.source/'delivery', destination)
        baseline = [r for r in read_rows(destination/'results.jsonl') if r['condition'] == 'neutral']
        if ({r['question_id'] for r in baseline} != set(plan['test_ids']) or
                any(r.get('prompt') != plan['prompt'] for r in baseline)):
            raise ValueError('Restore all 718 existing neutral test answers with the unchanged evaluation prompt before collection.')
        if shutil.disk_usage(output).free < 600*2**20: raise RuntimeError('Need 600 MiB free for aggregate checkpoints.')
        with Path('outputs/jenny-task-vectors.lock').open('a') as lock:
            print('Waiting for shared DIFFA GPU lock and idle GPU.', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(): time.sleep(30)
            from extract_vectors import load_model
            model, tokenizer, features = load_model()
            happy.collect(plan, output, model, tokenizer, features)
            source = happy.descriptor(plan, output)
            job = happy.steering_plan(source, plan['target'], [0., 1.])
            happy.winners.freeze(destination/'run_config.json', job)
            happy.winners.generate_job(job, destination, model, tokenizer, features)
            happy.report_steering(destination, tokenizer)


if __name__ == '__main__':
    main()
