"""BBH seven-object deduction: neutral/angry TTS, collect 125 pairs, test steering on 125."""

import argparse
import fcntl
import hashlib
from pathlib import Path
import random
import subprocess
import sys
import time

import run_reasoning_transfer as reasoning
from run_bbh_logical200 import PROMPT as THREE_PROMPT

SOURCE_SHA256 = '2896c7e3482eea318dd37bcc370d24ec3cc91e8374c1784287b5dbd38a529e33'
PROMPT = THREE_PROMPT.replace('three choices A, B, C', 'seven choices A, B, C, D, E, F, G').replace(
    'correct option A, B, or C', 'correct option A, B, C, D, E, F, or G')


def split_questions(data):
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('Official BBH seven-object source checksum differs.')
    questions = reasoning.bbh_questions(data, objects='seven')
    random.Random(42).shuffle(questions)
    return questions[:125], questions[125:]


def prepare(source, output):
    source.mkdir(parents=True, exist_ok=True)
    path = source/'logical_deduction_seven_objects-source.json'
    if not path.exists():
        path.write_bytes(reasoning.download(f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{reasoning.BBH_REVISION}/bbh/logical_deduction_seven_objects.json'))
    train, test = split_questions(path.read_bytes())
    questions = train+test
    reasoning.generate_audio(source, questions, ['neutral', 'angry'])
    bank = {q['question_id']: q for q in questions}
    records = reasoning.source_records(source, bank, ['neutral', 'angry'])
    return dict(task='bbh7', prompt=PROMPT, collection_prompt=PROMPT,
        vector=str((output/'style-vectors.pt').absolute()), bank={q['question_id']: q for q in test},
        source_sha256=SOURCE_SHA256,
        extraction=[dict(question_id=q['question_id'], neutral=records[q['question_id'], 'neutral'],
                         angry=records[q['question_id'], 'angry']) for q in train],
        recordings=reasoning.tasks_for(records, [q['question_id'] for q in test]),
        seed=42, max_new_tokens=512, steps=128, block_length=32, temperature=0., decoder='fast_dllm',
        layer=31, group='answer', strength=1,
        audio_encoding='Native up to 30s; full unchanged longer audio uses 30s chunked Whisper and global adapters',
        selection='Fixed seed42 125/125 split before seven-object outcomes; no outcome or classifier filtering',
        status_note='Prospectively fixed seven-object split/settings, motivated by prior three/five-object experiments.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['all', 'summary'], default='all')
    p.add_argument('--source', type=Path, default=Path('outputs/azure-bbh-seven250'))
    p.add_argument('--output', type=Path, default=Path('outputs/bbh-seven250-angry-mean'))
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        reasoning.report(args.output); return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scorer = reasoning.HarnessScorer(args.output/'harness/upstream', objects='seven')
        docs = [dict(input='Which?\nOptions:\n'+'\n'.join(f'({c}) Object {c}' for c in 'ABCDEFG'),
                     target=f'({letter})') for letter in 'ABCDEFG']
        if not all(r['correct'] for results in scorer.score([f'The answer is ({c}).' for c in 'ABCDEFG'], docs).values() for r in results):
            raise ValueError('Seven-object final-answer format does not match the upstream task filters.')
        plan = prepare(args.source, args.output)
        reasoning.freeze(args.output/'run_config.json', plan)
        print('125 collection / 125 test questions; own seven-object mean; L31 answer x1; seed42.', flush=True)
        print('Up to 250 collection + 375 evaluation generations (neutral/angry/steered-neutral).', flush=True)
        # TTS may run on CPU alongside existing work; inference starts after the earlier queues exit.
        for kind in ('reasoning-transfer', 'mean-steering', 'strength-sweep'):
            subprocess.run([sys.executable, '-u', 'wait_for_experiments.py', '--kind', kind], check=True)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            reasoning.evaluate(plan, args.output)
        reasoning.report(args.output)
        print(plan['status_note'], flush=True)


if __name__ == '__main__':
    main()
