"""GSM-only Final answer protocol; preserve the previous #### experiment."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import statistics
import subprocess
import time

import diffa_happy_nightly as happy
from scoring import SCORER_VERSION, score_response
from steer import read_rows


def report(rows, output, expected):
    groups = {}
    for row in rows:
        groups.setdefault(row['condition'], {})[row['question_id']] = row
    labels = ['neutral', 'happy', 'angry']
    labels += sorted(set(groups) - set(labels))
    active = [label for label in labels if groups.get(label)]
    common = set.intersection(*(set(groups[label]) for label in active)) if active else set()
    print('Saved:', ', '.join(f'{c}={len(groups.get(c, {}))}' for c in labels), flush=True)
    print(f'{len(common)}/{expected} matched questions across saved conditions; original {SCORER_VERSION}.', flush=True)
    scores = {}; summary = dict(expected=expected, matched=len(common), scorer=SCORER_VERSION, conditions={}, comparisons={})
    for label in active:
        selected = [groups[label][q] for q in sorted(common)]
        scores[label] = [score_response(r['response'], r['reference_answer'])['correct'] for r in selected]
        n = len(selected); correct = sum(scores[label])
        counts = [r['response_tokens'] for r in selected if r.get('response_tokens') is not None]
        info = dict(n=n, correct=correct, accuracy=correct/n if n else None,
                    mean_tokens=statistics.mean(counts) if counts else None,
                    median_tokens=statistics.median(counts) if counts else None)
        summary['conditions'][label] = info
        if n: print(f'{label}: {correct}/{n} ({correct/n:.1%}); mean/median tokens={info["mean_tokens"]}/{info["median_tokens"]}', flush=True)
    if common and 'neutral' in scores:
        from rescore_audio_steering import paired
        for label in active:
            if label == 'neutral': continue
            result = paired(list(zip(scores['neutral'], scores[label])))
            summary['comparisons'][label] = result
            print(f'{label} vs neutral: gain={result["gain_pp"]:+.2f}; p={result["mcnemar_exact_p"]:.4g}; CI={result["ci95_gain_pp"]}', flush=True)
    happy.winners.atomic_text(output/'summary.json', json.dumps(summary, indent=2)+'\n')


def prepare(source, output):
    original = json.loads((source/'run_config.json').read_text())
    plan = deepcopy(original)
    if plan['name'] != 'gsm8k' or 'Final answer: <number>' not in plan['collection_prompt']:
        raise ValueError('Expected the original GSM Final answer collection protocol.')
    plan['prompt'] = plan['collection_prompt']
    plan['target']['prompt'] = plan['prompt']
    config = json.loads(Path(plan['target']['config_file']).read_text())
    angry = {r['question_id']: r for r in config['recordings'] if r['condition'] == 'angry'}
    plan['recordings'] += [dict(angry[q], reference_answer=plan['bank'][q]['answer']) for q in plan['test_ids']]
    plan['protocol'] = 'diffa-gsm-final-answer-v1'
    plan['source_collection_config_sha256'] = happy.winners.digest(source/'run_config.json')
    output.mkdir(parents=True, exist_ok=True)
    happy.winners.freeze(output/'run_config.json', plan)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['run', 'summary', 'collection-summary'], default='run')
    p.add_argument('--source', type=Path, default=Path('outputs/diffa-happy-winners/gsm8k'))
    p.add_argument('--output', type=Path, default=Path('outputs/diffa-gsm-final-answer'))
    args = p.parse_args()
    if args.mode == 'collection-summary':
        from diffa_collection_report import collection_rows
        plan = json.loads((args.source/'run_config.json').read_text())
        destination = args.source/'collection-final-answer-report'; destination.mkdir(exist_ok=True)
        rows = collection_rows(plan, args.source)
        historical = {}
        for path in sorted(Path('outputs').rglob('results-matched-accuracy.jsonl')):
            for row in read_rows(path):
                if (row.get('question_id') in plan['collection_ids'] and row.get('voice') == happy.VOICE
                        and row.get('style') == 'angry' and row.get('prompt') == plan['collection_prompt']
                        and row.get('decoder') == plan['decoder'] and row.get('seed') == plan['seed']
                        and row.get('max_new_tokens') == plan['max_new_tokens']
                        and not row.get('interventions')
                        and str(row.get('reference_answer')) == str(plan['bank'][row['question_id']]['answer'])):
                    qid = row['question_id']
                    if qid in historical and historical[qid]['response'] != row['response']:
                        raise ValueError(f'Ambiguous archived angry answers for {qid}.')
                    historical[qid] = dict(row, condition='angry', reused_from=str(path))
        rows += list(historical.values())
        report(rows, destination, len(plan['collection_ids']))
        print('Neutral/happy: current collection. Angry: archived Final answer responses. Collection fitting questions, not held-out evidence.', flush=True)
        return
    if args.mode == 'summary':
        if not (args.output/'run_config.json').exists():
            print('Final answer run not prepared yet.'); return
        plan = json.loads((args.output/'run_config.json').read_text())
        report(read_rows(args.output/'results.jsonl'), args.output, len(plan['test_ids']))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'queue.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(args.source, args.output)
        # This vector was already collected with precisely the requested prompt.
        source = happy.descriptor(json.loads((args.source/'run_config.json').read_text()), args.source)
        job = happy.steering_plan(source, plan['target'], [1.])
        happy.winners.freeze(args.output/'steering-config.json', job)
        with Path('outputs/jenny-task-vectors.lock').open('a') as lock:
            print('Waiting for shared DIFFA GPU lock and idle GPU.', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                time.sleep(30)
            from extract_vectors import load_model
            model, tokenizer, features = load_model()
            done = {(r['question_id'], r['condition']) for r in read_rows(args.output/'results.jsonl', repair=True)}
            with (args.output/'results.jsonl').open('a') as stream:
                for record in plan['recordings']:
                    key = record['question_id'], record['condition']
                    if key in done: continue
                    print(f'Final answer {record["question_id"]} {record["condition"]}', flush=True)
                    response, encoding = happy.generate(record, plan, plan['prompt'], model, tokenizer, features)
                    saved = dict(record, response=response, prompt=plan['prompt'], seed=42,
                                 max_new_tokens=plan['max_new_tokens'], steps=128, block_length=32,
                                 decoder='fast_dllm', temperature=0., interventions=[], audio_encoding=encoding,
                                 response_tokens=len(tokenizer.encode(response, add_special_tokens=False)))
                    stream.write(json.dumps(saved)+'\n'); stream.flush(); done.add(key)
            happy.winners.generate_job(job, args.output, model, tokenizer, features)
            rows = read_rows(args.output/'results.jsonl')
            for row in rows:
                if row.get('response_tokens') is None:
                    row['response_tokens'] = len(tokenizer.encode(row['response'], add_special_tokens=False))
            happy.winners.atomic_text(args.output/'results.jsonl', ''.join(json.dumps(r)+'\n' for r in rows))
            report(rows, args.output, len(plan['test_ids']))


if __name__ == '__main__':
    main()
