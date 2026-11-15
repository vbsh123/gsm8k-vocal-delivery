"""Finish all 250 tracking questions, then collect on 125 and steer the other 125."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import run_bbh_reasoning200 as delivery
import steer_bbh_mean as steering
from run_reasoning_transfer import freeze
from steer import read_rows

TASK = 'tracking_shuffled_objects_three_objects'


def prepare(args):
    previous = args.existing_source/TASK
    args.source.mkdir(parents=True, exist_ok=True)
    data = (previous/'bbh-source.json').read_bytes()
    selected = delivery.questions(data, TASK, 250)
    if read_rows(previous/'questions.jsonl') != selected[:200]:
        raise ValueError('Existing recordings differ from the first 200 selected questions.')
    (args.source/'bbh-source.json').write_bytes(data)
    (args.source/'questions.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in selected))
    command = [sys.executable, '-u', 'cloud_tts_smoke.py', '--provider', 'azure',
        '--azure-voice', 'en-US-JennyNeural', '--azure-intensity', '1.5',
        '--input-jsonl', str(args.source/'questions.jsonl'), '--num-examples', '250',
        '--styles', *delivery.STYLES, '--output', str(args.source), '--resume', '--retry-passes', '5']
    subprocess.run(command+['--prepare-only'], check=True)
    old = {(r['question_id'], r['voice'], r['style']): r for r in read_rows(previous/'plan.jsonl')}
    for row in read_rows(args.source/'plan.jsonl'):
        existing = old.get((row['question_id'], row['voice'], row['style']))
        if existing is None:
            continue
        if any(row[k] != existing[k] for k in ('question', 'answer', 'request')):
            raise ValueError('Existing audio uses different wording or TTS settings.')
        target = args.source/row['audio_path']; source = previous/existing['audio_path']
        if source.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source.absolute())
    print('Reusing the original 600 clips; generating only missing clips from the last 50 questions.', flush=True)
    return delivery.prepare(SimpleNamespace(source=args.source.parent, num_examples=250), TASK)


def copy_answers(source, output, plan, allowed=None):
    """Reuse compatible generations; the saved waveform fingerprint must match."""
    tasks = {(r['question_id'], r['voice'], r['style']): r for r in plan['recordings']}
    done = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(output/'results.jsonl')}
    with (output/'results.jsonl').open('a') as stream:
        for row in read_rows(source/'results.jsonl'):
            key = row['question_id'], row['voice'], row['condition']
            if key in done or key not in tasks or (allowed is not None and
                    (key[0] not in allowed or key[2] not in ('neutral', 'angry'))):
                continue
            task = tasks[key]
            if (row['prompt'] != plan['prompt'] or row['seed'] != 42 or row['max_new_tokens'] != 512
                    or row['decoder'] != 'fast_dllm' or row['audio_sha256'] != task['audio_sha256']
                    or row['reference_answer'] != task['answer']):
                raise ValueError('Existing answer differs from the tracking recipe.')
            stream.write(json.dumps(dict(row, reused_from=str(source)))+'\n')
            done.add(key)


def split_plan(plan, source):
    audio_paths = {(r['question_id'], r['voice'], r['style']): r['audio_path']
                   for r in read_rows(source/'plan.jsonl')}
    groups = {}
    for row in plan['recordings']:
        if row['style'] in ('neutral', 'angry'):
            key = row['question_id'], row['voice'], row['style']
            groups.setdefault(row['question_id'], {})[row['style']] = dict(row, audio_path=audio_paths[key])
    pairs = [dict(question_id=q, **groups[q]) for q in plan['bank']]
    return dict(extraction=pairs[:125], test=pairs[125:], prompt=plan['prompt'],
        harness_task=plan['harness_task'], seed=42, max_new_tokens=512, steps=128,
        block_length=32, decoder='fast_dllm', temperature=0., layer=31, group='answer', strength=1,
        audio_encoding=dict(mode='chunked-whisper-global-adapters-v1', chunk_seconds=30),
        averaging='Mean angry-minus-neutral over matched forwards per question, then equal weight per question')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['all', 'delivery', 'steering', 'summary'], default='all')
    p.add_argument('--existing-source', type=Path, default=Path('outputs/azure-bbh-reasoning200'))
    p.add_argument('--existing-output', type=Path, default=Path('outputs/bbh-reasoning200-diffa2'))
    p.add_argument('--source', type=Path, default=Path('outputs/azure-bbh-reasoning250')/TASK)
    p.add_argument('--output', type=Path, default=Path('outputs/bbh-tracking250'))
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    benchmark = args.output/'deliveries'; benchmark.mkdir(exist_ok=True)
    collected = args.output/'steering'; collected.mkdir(exist_ok=True)
    worker = SimpleNamespace(source=args.source, output=collected)
    if args.stage == 'summary':
        delivery.report(benchmark)
        if (collected/'split.json').exists():
            steering.summarize(worker)
        return
    # Separate children release the delivery model before loading it for collection.
    if args.stage == 'all':
        with (args.output/'queue.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for stage in ('delivery', 'steering'):
                print('STAGE:', stage, flush=True)
                subprocess.run([sys.executable, '-u', str(Path(__file__).absolute()),
                                *sys.argv[1:], '--stage', stage], check=True)
        return
    if args.stage == 'delivery':
        plan = prepare(args)
        freeze(benchmark/'run_config.json', plan)
        copy_answers(args.existing_output/TASK, benchmark, plan)
    else:
        plan = json.loads((benchmark/'run_config.json').read_text())
        split = split_plan(plan, args.source)
        freeze(collected/'split.json', split)
        copy_answers(benchmark, collected, plan, {r['question_id'] for r in split['test']})
    with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
        print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            print('Waiting for active GPU work.', flush=True); time.sleep(30)
        if args.stage == 'delivery':
            delivery.run(plan, benchmark)
        else:
            steering.run(worker, split)
    if args.stage == 'delivery':
        delivery.report(benchmark)
    else:
        steering.summarize(worker)


if __name__ == '__main__':
    main()
