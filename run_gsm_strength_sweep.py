"""GSM8K L31 answer strength sweep, -2 through 2 by 0.5, after the current queues."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import run_reasoning_transfer as reasoning
from steer import read_rows

STRENGTHS = tuple(i/2 for i in range(-4, 5))


def make_plan(previous):
    if previous['task'] != 'gsm8k' or previous['prompt'] != reasoning.HASH_PROMPT:
        raise ValueError('Expected the aligned-prompt GSM8K evaluation, not the older Final answer prompt.')
    records = [r for r in previous['recordings'] if r['condition'] == 'neutral']
    tasks = [dict(r, strength=a, condition='neutral' if a == 0 else f'neutral-L31-answer-a{a:g}')
             for r in records for a in STRENGTHS]
    return dict(previous, recordings=tasks, strengths=list(STRENGTHS),
                selection='Same GSM8K evaluation questions and collected vector; exploratory strength sensitivity sweep')


def reuse_answers(benchmark, output, plan):
    tasks = {(r['question_id'], r['voice'], r['condition']): r for r in plan['recordings']}
    done = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(output/'results.jsonl', repair=True)}
    with (output/'results.jsonl').open('a') as stream:
        for row in read_rows(benchmark/'results.jsonl'):
            key = row['question_id'], row['voice'], row['condition']
            if key not in tasks or key in done:
                continue
            task = tasks[key]
            if (row['prompt'] != plan['prompt'] or row['seed'] != plan['seed']
                    or row['max_new_tokens'] != plan['max_new_tokens'] or row['decoder'] != plan['decoder']
                    or row['audio_sha256'] != task['audio_sha256'] or row['reference_answer'] != task['answer']):
                raise ValueError('Existing aligned-prompt answer differs from the sweep recipe.')
            expected = [dict(layer=31, group='answer', strength=task['strength'])] if task['strength'] else []
            if row['interventions'] != expected:
                raise ValueError('Saved answer uses different steering settings.')
            stream.write(json.dumps(dict(row, reused_from=str(benchmark)))+'\n')
            done.add(key)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['all', 'summary'], default='all')
    p.add_argument('--benchmark', type=Path, default=Path('outputs/reasoning-transfer/gsm8k'))
    p.add_argument('--output', type=Path, default=Path('outputs/gsm8k-L31-strength-sweep'))
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        reasoning.report(args.output); return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for kind in ('reasoning-transfer', 'mean-steering'):
            subprocess.run([sys.executable, '-u', 'wait_for_experiments.py', '--kind', kind], check=True)
        previous = json.loads((args.benchmark/'run_config.json').read_text())
        plan = make_plan(previous)
        reasoning.freeze(args.output/'run_config.json', plan)
        reuse_answers(args.benchmark, args.output, plan)
        done = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(args.output/'results.jsonl')}
        pending = sum((r['question_id'], r['voice'], r['condition']) not in done for r in plan['recordings'])
        print('L31 answer strengths:', STRENGTHS, flush=True)
        print(f'{len(plan["bank"])} questions; {pending} generations pending; strengths 0 and +1 reused when complete.', flush=True)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            reasoning.evaluate(plan, args.output)
        reasoning.report(args.output)


if __name__ == '__main__':
    main()
