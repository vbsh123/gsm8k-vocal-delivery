"""Dataset emotion audit and separately selected DIFFA angry-vector experiments."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import audit_available_audio as audit
import diffa_happy_nightly as collection
import winning_diffa_sweeps as winners
from steer import read_rows

DATASETS = ('gsm8k', 'svamp', 'multiarith', 'bbh3', 'bbh5', 'ifeval')
STYLES = ('neutral', 'angry', 'happy')
MODEL = 'emotion2vec/emotion2vec_plus_large'


def audio_candidates(root):
    found = {}
    for path in sorted(root.rglob('plan.jsonl')):
        for row in read_rows(path):
            if (not isinstance(row, dict) or not row.get('question_id') or not row.get('audio_path')
                    or row.get('voice') != collection.VOICE or row.get('provider') != 'azure'
                    or row.get('style') not in STYLES or not row.get('question')): continue
            if not (path.parent/row['audio_path']).is_file(): continue
            found.setdefault((row['question_id'], row['style']), []).append((row, path.parent))
    return found


def inventory(output, names, catalog, candidates):
    """Keep every surviving relevant clip; pin original neutral/angry recordings."""
    output.mkdir(parents=True, exist_ok=True)
    datasets = {}; clips = {}; issues = []
    for name in names:
        target = winners.load_dataset(name, catalog[name])
        config = json.loads(Path(target['config_file']).read_text())
        train = collection.historical_collection_ids(target)
        test = [r['question_id'] for r in target['records']]
        if set(train) & set(test): raise ValueError('Collection/test overlap.')
        prefixes = {re.sub(r'[_-]?\d+$', '', q) for q in train+test}
        other = sorted({q for q, _ in candidates if re.sub(r'[_-]?\d+$', '', q) in prefixes} - set(train+test))
        expected = {(r['question_id'], 'neutral'): r for r in target['records']}
        for pair in config.get('extraction', []):
            for style in ('neutral', 'angry'):
                expected[pair['question_id'], style] = pair[style]
        for r in config.get('recordings', []):
            if r.get('condition') in ('neutral', 'angry'):
                expected[r['question_id'], r['condition']] = r
        records = []; pinned = {}
        for split, ids in [('collection', train), ('test', test), ('other', other)]:
            for qid in ids:
                for style in STYLES:
                    wanted = expected.get((qid, style), {}).get('audio_sha256')
                    options = []
                    for row, folder in candidates.get((qid, style), []):
                        record = collection.make_record(row, folder)
                        key = record['audio_sha256']
                        previous = expected.get((qid, style), {})
                        gold = previous.get('reference_answer', previous.get('answer'))
                        if (wanted == key and gold is not None and
                                record['reference_answer'] != collection.normal_answer(gold)):
                            raise ValueError(f'Original reference changed: {name}/{qid}/{style}.')
                        alias = dict(dataset=name, split=split, question_id=qid, requested_style=style,
                                     voice=collection.VOICE, question=row['question'], answer=record['reference_answer'])
                        clip = clips.setdefault(key, dict(clip_id=key, audio_sha256=key,
                            audio_file=record['audio_file'], paths=[], aliases=[]))
                        if record['audio_file'] not in clip['paths']: clip['paths'].append(record['audio_file'])
                        if alias not in clip['aliases']: clip['aliases'].append(alias)
                        records.append(dict(alias, **{'audio_sha256': key}))
                        if not wanted or key == wanted:
                            options.append((row, record))
                    if options:
                        # Prefer the exact original hash; ambiguity is never silently resolved.
                        hashes = {r['audio_sha256'] for _, r in options}
                        if len(hashes) > 1 and style in ('neutral', 'angry'):
                            raise ValueError(f'Ambiguous original recording: {name}/{qid}/{style}.')
                        row, record = options[0]
                        pinned[qid+'|'+style] = dict(record=record, question=row['question'])
                    else:
                        issues.append(dict(dataset=name, split=split, question_id=qid, style=style,
                                           reason='Missing original audio' if wanted else 'No generated audio available'))
        datasets[name] = dict(target=target, collection_ids=train, test_ids=test, other_ids=other, pinned=pinned,
                              recordings=records)
    cohorts = dict(protocol='diffa-angry-agreed-audit-v1', datasets=datasets, issues=issues)
    winners.atomic_text(output/'cohorts.json', json.dumps(cohorts, indent=2)+'\n')
    winners.atomic_text(output/'audio_inventory.jsonl', ''.join(json.dumps(r)+'\n' for _, r in sorted(clips.items())))
    seed_labels(output)
    return cohorts


def seed_labels(output):
    """Reuse only whole-recording predictions keyed by verified audio SHA256."""
    wanted = {r['clip_id'] for r in read_rows(output/'audio_inventory.jsonl')}
    path = output/'results-emotion2vec.jsonl'
    saved = {r['clip_id']: r for r in read_rows(path, repair=True)}
    for source in sorted(Path('outputs').rglob('results-emotion2vec.jsonl')):
        if source.resolve() == path.resolve(): continue
        for row in read_rows(source):
            key = row.get('audio_sha256')
            if (key not in wanted or key in saved or row.get('model') != MODEL
                    or row.get('processing_mode') != 'whole_recording'
                    or row.get('classification_status') != 'classified'): continue
            saved[key] = dict(row, clip_id=key, reused_from=str(source.absolute()))
    winners.atomic_text(path, ''.join(json.dumps(r)+'\n' for r in saved.values()))


def predictions(output):
    return {r['clip_id']: r for r in read_rows(output/'results-emotion2vec.jsonl')}


def agreed_pairs(dataset, labels):
    selected = []
    for qid in dataset['collection_ids']:
        pair = {}; ok = True
        for style, key in [('neutral', 'neutral'), ('angry', 'emotion')]:
            item = dataset['pinned'].get(qid+'|'+style)
            label = labels.get(item['record']['audio_sha256'], {}) if item else {}
            if label.get('classification_status') != 'classified' or label.get('emotion') != style:
                ok = False; break
            pair[key] = item['record']
        if ok:
            if dataset['pinned'][qid+'|neutral']['question'] != dataset['pinned'][qid+'|angry']['question']:
                raise ValueError('Paired transcripts differ.')
            if pair['neutral']['reference_answer'] != pair['emotion']['reference_answer']:
                raise ValueError('Paired references differ.')
            selected.append(dict(question_id=qid, **pair))
    return selected


def summarize_audit(output):
    cohorts = json.loads((output/'cohorts.json').read_text()); labels = predictions(output)
    rows = []; eligibility = {}
    print('Dataset       Split       Style     Expected Available Classified Agreed Match% Errors Pending', flush=True)
    for name, data in cohorts['datasets'].items():
        splits = [('collection', data['collection_ids']), ('test', data['test_ids'])]
        if data.get('other_ids'): splits.append(('other', data['other_ids']))
        splits.insert(0, ('all', [q for _, ids in splits for q in ids]))
        for split, ids in splits:
            for style in STYLES:
                # All surviving variants, deduplicated per dataset/split/style/audio hash.
                keys = {r['audio_sha256'] for r in data['recordings']
                        if (split == 'all' or r['split'] == split) and r['requested_style'] == style}
                valid = [labels[k] for k in keys if labels.get(k, {}).get('classification_status') == 'classified']
                n = len(valid); agreed = sum(r.get('emotion') == style for r in valid)
                errors = sum(labels.get(k, {}).get('classification_status') == 'error' for k in keys)
                row = dict(dataset=name, split=split, style=style, expected_questions=len(ids), available_audio=len(keys),
                           classified=n, agreed=agreed, match_pct=100*agreed/n if n else None,
                           errors=errors, pending=len(keys)-n-errors, predictions=dict(Counter(r['emotion'] for r in valid)))
                rows.append(row)
                pct = f'{row["match_pct"]:.1f}' if n else 'n/a'
                print(f'{name:13} {split:11} {style:8} {len(ids):8} {len(keys):9} {n:10} {agreed:6} {pct:>6} {errors:6} {row["pending"]:7}', flush=True)
        selected = agreed_pairs(data, labels)
        eligibility[name] = dict(collection_pool=len(data['collection_ids']), agreed_pairs=len(selected), test_questions=len(data['test_ids']))
        print(f'  {name}: {len(selected)}/{len(data["collection_ids"])} neutral/angry-agreed collection pairs; {len(data["test_ids"])} unchanged test questions.', flush=True)
    summary = dict(rows=rows, eligibility=eligibility, issues=cohorts['issues'],
                   note='Emotion2vec is a model-based selection rule, not human ground truth. No test filtering or correctness selection.')
    winners.atomic_text(output/'dataset-summary.json', json.dumps(summary, indent=2)+'\n')
    with (output/'dataset-summary.csv').open('w', newline='') as stream:
        fields = [k for k in rows[0] if k != 'predictions']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    return summary


def prepare_plan(name, data, labels, audit_root, folder):
    target = deepcopy(data['target']); pairs = agreed_pairs(data, labels)
    if not pairs: raise ValueError(f'{name}: no agreed collection pairs. Inspect the audit first.')
    # Do not interpret missing predictions as disagreement: finish the audit before selecting.
    for qid in data['collection_ids']:
        for style in ('neutral', 'angry'):
            item = data['pinned'].get(qid+'|'+style)
            if not item: raise ValueError(f'{name}: missing original collection audio {qid}/{style}.')
            label = labels.get(item['record']['audio_sha256'], {})
            if label.get('classification_status') != 'classified':
                raise ValueError(f'{name}: collection label pending/error for {qid}/{style}; finish or retry audit.')
    bank = dict(target['bank'])
    for pair in pairs:
        q = pair['question_id']; record = pair['neutral']
        bank[q] = dict(question_id=q, question=data['pinned'][q+'|neutral']['question'], answer=record['reference_answer'])
    receipts = [dict(question_id=p['question_id'], labels={style: labels[p[key]['audio_sha256']]
                 for style, key in [('neutral', 'neutral'), ('angry', 'emotion')]}) for p in pairs]
    plan = dict(protocol='diffa-angry-agreed-v1', name=name, emotion='angry', target=target,
                source_run=target['run'], collection_ids=[p['question_id'] for p in pairs], test_ids=data['test_ids'],
                pairs=pairs, bank=bank, prompt=target['prompt'], collection_prompt=target['vector_info']['collection_prompt'],
                seed=42, max_new_tokens=target['max_new_tokens'], steps=128, block_length=32, decoder='fast_dllm', temperature=0.,
                averaging='Matched forwards within each pair; token-position means; equal per-question weight.',
                classifier=MODEL, classifier_receipts=receipts,
                collection_pool_ids=data['collection_ids'], audit_manifest_sha256=winners.digest(audit_root/'cohorts.json'),
                selection='Filter within the original collection half only: neutral and angry both recognized; test split unchanged.')
    folder.mkdir(parents=True, exist_ok=True); winners.freeze(folder/'run_config.json', plan)
    return plan


def descriptor(plan, folder):
    result = dict(plan['target'], emotion='angry', vector=str((folder/'style-vectors.pt').absolute()),
                  vector_sha256=winners.digest(folder/'style-vectors.pt'), extraction_ids=plan['collection_ids'])
    _, result['vector_info'] = winners.read_vector(result['vector'], 'angry')
    return result


def job_for(source, target, strengths):
    if set(source['extraction_ids']) & {r['question_id'] for r in target['records']}:
        raise ValueError('Collection overlaps transfer/evaluation test IDs.')
    return dict(protocol='diffa-angry-agreed-v1', source=source, target=target, strengths=strengths,
                vector=source['vector'], vector_sha256=source['vector_sha256'],
                layer=31, group='answer', schedule='always', mode='fixed')


def prepare_job(job, folder):
    folder.mkdir(parents=True, exist_ok=True); winners.freeze(folder/'run_config.json', job)
    winners.reuse_answers(job, folder)
    baseline = [r for r in read_rows(folder/'results.jsonl') if r['condition'] == 'neutral']
    if {r['question_id'] for r in baseline} != {r['question_id'] for r in job['target']['records']}:
        raise ValueError('Missing compatible neutral baseline; no new baseline generation authorized.')
    winners.score_rows(job, baseline, folder)  # Evaluator preflight before GPU loading.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['audit', 'inventory', 'audit-summary', 'prepare', 'run', 'summary'], default='audit-summary')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS[1:]))
    p.add_argument('--stage', choices=['primary', 'strengths', 'transfer'], default='primary')
    p.add_argument('--strengths', nargs='+', type=float, default=list(winners.STRENGTHS))
    p.add_argument('--catalog', type=Path, default=Path('winning_diffa_catalog.json'))
    p.add_argument('--output', type=Path, default=Path('outputs/diffa-angry-agreed'))
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if not args.strengths or not all(math.isfinite(x) for x in args.strengths): p.error('Strengths must be finite.')
    root = args.output; audit_root = root/'audit'; root.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        for name in args.datasets:
            folder = root/name/args.stage
            if (folder/'run_config.json').exists(): collection.report_steering(folder)
            else: print(f'{name}/{args.stage}: not run/prepared.')
        if args.stage == 'transfer':
            for folder in sorted((root/'transfer').glob('*')): collection.report_steering(folder)
        return
    if args.mode == 'audit-summary': summarize_audit(audit_root); return
    with (root/'queue.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode in ('inventory', 'audit'):
            inventory(audit_root, args.datasets, json.loads(args.catalog.read_text()), audio_candidates(Path('outputs')))
            summarize_audit(audit_root)
            if args.mode == 'inventory': return
            saved = predictions(audit_root)
            pending = [r for r in read_rows(audit_root/'audio_inventory.jsonl')
                       if saved.get(r['clip_id'], {}).get('classification_status') != 'classified']
            if pending:
                python = Path(os.environ.get('EMOTION_PYTHON', '.venv-emotion/bin/python')).absolute()
                if not python.is_file(): raise FileNotFoundError(f'Emotion environment missing: {python}; set EMOTION_PYTHON.')
                command = [str(python), '-u', 'audit_available_audio.py', '--mode', 'classify', '--model', 'emotion2vec',
                           '--output', str(audit_root), '--device', args.device, '--retry-errors']
                if args.device.startswith('cuda'):
                    with Path('outputs/jenny-task-vectors.lock').open('a') as lock:
                        print('Waiting for shared GPU queue for emotion2vec.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
                        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(): time.sleep(30)
                        subprocess.run(command, check=True)
                else: subprocess.run(command, check=True)
            summarize_audit(audit_root); return
        cohorts = json.loads((audit_root/'cohorts.json').read_text())['datasets']; labels = predictions(audit_root)
        plans = {}
        for name in args.datasets:
            if name not in cohorts: raise ValueError(f'Audit {name} first.')
            folder = root/name
            if (folder/'run_config.json').exists():
                plan = json.loads((folder/'run_config.json').read_text())
                if plan['audit_manifest_sha256'] != winners.digest(audit_root/'cohorts.json'):
                    raise ValueError('Audit inventory changed after selection; use a separate output root.')
            else: plan = prepare_plan(name, cohorts[name], labels, audit_root, folder)
            target = plan['target']
            if (winners.digest(target['config_file']) != target['config_sha256'] or
                    winners.digest(target['vector']) != target['vector_sha256']):
                raise ValueError('Original experiment receipts changed after audit.')
            plans[name] = plan
            if args.stage == 'primary':
                # Use original source solely for baseline preflight; replace the direction after collection.
                preflight = job_for(plan['target'], plan['target'], [0.])
                prepare_job(preflight, folder/'baseline')
        if args.mode == 'prepare': return
        if shutil.disk_usage(root).free < 600*2**20: raise RuntimeError('Need at least 600 MiB free for streaming collection.')
        if args.stage != 'primary':
            for name in plans:
                if not (root/name/'style-vectors.pt').exists(): raise ValueError('Complete primary vector collection first.')
        future_jobs = []
        if args.stage != 'primary':
            sources = {name: descriptor(plan, root/name) for name, plan in plans.items()}
            for source_name, source in sources.items():
                targets = plans.items() if args.stage == 'transfer' else [(source_name, plans[source_name])]
                for target_name, plan in targets:
                    if args.stage == 'transfer' and source_name == target_name: continue
                    target = deepcopy(plan['target'])
                    target['reuse_runs'] += [str((root/target_name/'primary').absolute())]
                    strengths = [0., 1.] if args.stage == 'transfer' else list(dict.fromkeys([0., *args.strengths]))
                    job = job_for(source, target, strengths)
                    folder = (root/'transfer'/f'{target_name}__from__{source_name}' if args.stage == 'transfer'
                              else root/source_name/'strengths')
                    prepare_job(job, folder); future_jobs.append((job, folder))
        with Path('outputs/jenny-task-vectors.lock').open('a') as lock:
            print('Waiting for shared DIFFA GPU lock and idle GPU.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(): time.sleep(30)
            from extract_vectors import load_model
            model, tokenizer, features = load_model()
            winners.atomic_text(root/'runtime.json', json.dumps(dict(
                runner_sha256=winners.digest(__file__), collector_sha256=winners.digest(collection.__file__),
                stage=args.stage, datasets=args.datasets, model_config=model.config.to_dict()), indent=2, default=str)+'\n')
            if args.stage == 'primary':
                for name, plan in plans.items():
                    folder = root/name
                    collection.collect(plan, folder, model, tokenizer, features)
                    source = descriptor(plan, folder)
                    target = deepcopy(plan['target'])
                    target['reuse_runs'] += [str((folder/'primary').absolute())]
                    job = job_for(source, target, [0., 1.]); destination = folder/'primary'
                    prepare_job(job, destination)
                    winners.generate_job(job, destination, model, tokenizer, features)
                    collection.report_steering(destination, tokenizer)
            else:
                for job, folder in future_jobs:
                    winners.generate_job(job, folder, model, tokenizer, features)
                    collection.report_steering(folder, tokenizer)


if __name__ == '__main__': main()
