"""Collect an angry-minus-neutral IFEval mean on half the prompts; steer the other half."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import run_spoken_ifeval as ifeval
import steer_bbh_mean as collection
from run_reasoning_transfer import freeze
from steer import read_rows

CONDITION = 'neutral-IFEval-L31-answer-a1'


def make_plan(source, benchmark):
    previous = json.loads((benchmark/'run_config.json').read_text())
    if (previous['prompt'] != ifeval.PROMPT or previous['seed'] != 42
            or previous['max_new_tokens'] != 1024 or previous['decoder'] != 'fast_dllm'
            or previous['revision'] != ifeval.REVISION):
        raise ValueError('Expected the original IFEval prompt, seed, scorer and 1024-token recipe.')
    # Preserve the earlier four-style length eligibility, rather than selecting a new angry/neutral subset.
    groups = {}
    for row in previous['recordings']:
        groups.setdefault(row['question_id'], {})[row['style']] = row
    if any(set(rows) != set(ifeval.STYLES) for rows in groups.values()):
        raise ValueError('Expected the previous matched four-style IFEval cohort.')
    ids = sorted(groups); random.Random(42).shuffle(ids)
    pairs = []
    for qid in ids:
        pair = dict(question_id=qid)
        for style in ('neutral', 'angry'):
            row = groups[qid][style]; path = (source/row['audio_path']).absolute()
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['audio_sha256']:
                raise ValueError('IFEval audio changed since the original evaluation.')
            pair[style] = dict(row, audio_file=str(path))
        if pair['neutral']['voice'] != pair['angry']['voice']:
            raise ValueError('Expected matched question and voice.')
        pairs.append(pair)
    half = len(pairs)//2
    if not half:
        raise ValueError('Need at least two eligible IFEval questions.')
    return dict(extraction=pairs[:half], test=pairs[half:], prompt=ifeval.PROMPT,
        seed=42, max_new_tokens=1024, steps=128, block_length=32, decoder='fast_dllm', temperature=0.,
        revision=ifeval.REVISION, layer=31, group='answer', strength=1, steered_condition=CONDITION,
        record_response_tokens=True, primary_metric='strict_prompt',
        audio_encoding=dict(mode='native', chunk_seconds=30),
        averaging='Mean angry-minus-neutral over matched forwards per question, then equal weight per question',
        selection='Seed42 shuffled halves of the previous four-style eligible cohort; no outcome/classifier filter',
        exploratory='Aggregate emotion-delivery results were inspected before selecting this steering follow-up.')


def reuse_answers(benchmark, output, plan):
    tasks = {(p['question_id'], p[s]['voice'], s): p[s] for p in plan['test'] for s in ('neutral', 'angry')}
    done = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(output/'results.jsonl', repair=True)}
    with (output/'results.jsonl').open('a') as stream:
        for row in read_rows(benchmark/'results.jsonl'):
            key = row['question_id'], row['voice'], row['style']
            if key not in tasks or key in done:
                continue
            if (row['wrapper'] != plan['prompt'] or row['seed'] != 42 or row['max_new_tokens'] != 1024
                    or row['audio_path'] != tasks[key]['audio_path']):
                raise ValueError('Existing IFEval answer differs from the steering recipe.')
            stream.write(json.dumps(dict(row, condition=row['style'], prompt=plan['prompt'],
                                         reused_from=str(benchmark)))+'\n')
            done.add(key)


def report(args):
    if not (args.output/'split.json').exists():
        print('IFEval steering: waiting / not prepared.', flush=True); return
    plan = json.loads((args.output/'split.json').read_text())
    allowed = {p['question_id'] for p in plan['test']}
    if any(r['question_id'] not in allowed for r in read_rows(args.output/'results.jsonl')):
        raise ValueError('Collection question found in held-out results.')
    print(f'IFEval: {len(plan["extraction"])} collection / {len(allowed)} test questions; L31 answer x1; seed42.')
    ifeval.report(args, styles=['neutral', 'angry', CONDITION], condition_field='condition')
    print(plan['exploratory'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['all', 'summary'], default='all')
    p.add_argument('--source', type=Path, default=Path('outputs/azure-spoken-ifeval'))
    p.add_argument('--benchmark', type=Path, default=Path('outputs/spoken-ifeval-diffa2'))
    p.add_argument('--output', type=Path, default=Path('outputs/ifeval-angry-mean'))
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        report(args); return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for kind in ('ifeval', 'ifbench', 'reasoning-transfer'):
            subprocess.run([sys.executable, '-u', 'wait_for_experiments.py', '--kind', kind], check=True)
        plan = make_plan(args.source, args.benchmark)
        freeze(args.output/'split.json', plan)
        reuse_answers(args.benchmark, args.output, plan)
        ifeval.setup_official(); ifeval.official_scorer()
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            collection.run(args, plan)
        report(args)


if __name__ == '__main__':
    main()
