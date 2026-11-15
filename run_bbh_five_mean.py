"""Collect a five-object BBH vector on 125 questions and test L31 x1 on the other 125."""

import argparse
import fcntl
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import run_reasoning_transfer as reasoning
from steer import read_rows


def prepare(source, benchmark, output):
    original = source/'logical_deduction_five_objects-source.json'
    questions = reasoning.bbh_questions(original.read_bytes())
    bank = {q['question_id']: q for q in questions}
    previous = json.loads((benchmark/'run_config.json').read_text())
    if (previous['task'] != 'bbh5' or previous['bank'] != bank
            or previous['source_sha256'] != reasoning.digest(original)
            or previous['prompt'] != reasoning.BBH_PROMPT or previous['seed'] != 42
            or previous['decoder'] != 'fast_dllm' or previous['max_new_tokens'] != 512):
        raise ValueError('Existing BBH-five run differs from the expected source/prompt/decoding recipe.')
    records = reasoning.source_records(source, bank, ['neutral', 'angry'])
    expected = {(r['question_id'], r['style']): r['audio_sha256'] for r in previous['recordings']}
    if any(expected[key] != row['audio_sha256'] for key, row in records.items()):
        raise ValueError('BBH-five audio changed since the earlier evaluation.')
    ids = list(bank); random.Random(42).shuffle(ids)
    train, test = ids[:125], ids[125:]
    plan = dict(previous, bank={q: bank[q] for q in test},
        vector=str((output/'style-vectors.pt').absolute()), collection_prompt=reasoning.BBH_PROMPT,
        extraction=[dict(question_id=q, neutral=records[q, 'neutral'], angry=records[q, 'angry']) for q in train],
        recordings=reasoning.tasks_for(records, test),
        selection='Fixed seed42 125/125 split; all pairs; no outcome or classifier filtering',
        exploratory='Five-object delivery and three-object-vector transfer results were inspected before this follow-up.')
    plan.pop('vector_sha256', None)
    return plan


def reuse_answers(benchmark, output, plan):
    tasks = {(r['question_id'], r['voice'], r['condition']): r for r in plan['recordings']
             if r['condition'] in ('neutral', 'angry')}
    done = {(r['question_id'], r['voice'], r['condition'])
            for r in read_rows(output/'results.jsonl', repair=True)}
    with (output/'results.jsonl').open('a') as stream:
        for row in read_rows(benchmark/'results.jsonl'):
            key = row['question_id'], row['voice'], row['condition']
            if key not in tasks or key in done:
                continue
            task = tasks[key]
            if (row['prompt'] != plan['prompt'] or row['seed'] != 42 or row['max_new_tokens'] != 512
                    or row['decoder'] != 'fast_dllm' or row['audio_sha256'] != task['audio_sha256']
                    or row['reference_answer'] != task['answer']):
                raise ValueError('Saved neutral/angry answer differs from this evaluation recipe.')
            stream.write(json.dumps(dict(row, reused_from=str(benchmark)))+'\n')
            done.add(key)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['all', 'summary'], default='all')
    p.add_argument('--source', type=Path, default=Path('outputs/azure-bbh-five250'))
    p.add_argument('--benchmark', type=Path, default=Path('outputs/reasoning-transfer/bbh5'))
    p.add_argument('--output', type=Path, default=Path('outputs/bbh-five250-angry-mean'))
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        reasoning.report(args.output); return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('Waiting for the current reasoning queue, including SVAMP and GSM8K, to exit.', flush=True)
        subprocess.run([sys.executable, '-u', 'wait_for_experiments.py', '--kind', 'reasoning-transfer'], check=True)
        plan = prepare(args.source, args.benchmark, args.output)
        reasoning.freeze(args.output/'run_config.json', plan)
        reuse_answers(args.benchmark, args.output, plan)
        print('125 extraction / 125 test questions; own five-object mean; L31 answer x1; seed42.', flush=True)
        print('Up to 250 collection generations + 125 steered generations; saved neutral/angry answers reused.', flush=True)
        reasoning.scorer_for(plan, args.output)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            reasoning.evaluate(plan, args.output)
        reasoning.report(args.output)
        print(plan['exploratory'], 'P-values and intervals are unadjusted.', flush=True)


if __name__ == '__main__':
    main()
