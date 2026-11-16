"""MultiArith: deduplicate, collect a task mean on half, evaluate on the other half."""

import argparse
from decimal import Decimal
import fcntl
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import run_reasoning_transfer as reasoning

SOURCE_REVISION = '5ef330fcdeec0cd26aee27943504f91f8ec1c33c'
SOURCE_URL = f'https://raw.githubusercontent.com/kojima-takeshi188/zero_shot_cot/{SOURCE_REVISION}/dataset/MultiArith/MultiArith.json'
SOURCE_SHA256 = 'dc9db1b6cc68b1ecc2dec830d15aaf8c9a0932aa31d5ffce88ef3628179d36f4'
PROMPT = reasoning.HASH_PROMPT


def split_questions(data):
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('MultiArith source checksum differs.')
    rows = json.loads(data)
    if len(rows) != 600 or len({r['iIndex'] for r in rows}) != 600:
        raise ValueError('Expected 600 distinct MultiArith source IDs.')
    unique = {}
    for row in rows:
        if len(row['lSolutions']) != 1:
            raise ValueError('Expected exactly one numeric MultiArith answer.')
        answer = Decimal(str(row['lSolutions'][0]))
        if not answer.is_finite():
            raise ValueError('Non-finite MultiArith answer.')
        answer = format(answer.normalize(), 'f')
        key = ' '.join(row['sQuestion'].split())
        if not key:
            raise ValueError('Empty MultiArith question.')
        if key in unique:
            if unique[key]['answer'] != answer:
                raise ValueError('Duplicate MultiArith question has conflicting answers.')
            unique[key]['source_ids'].append(row['iIndex'])
            continue
        unique[key] = dict(question_id=f'multiarith_{row["iIndex"]:03d}',
            source_ids=[row['iIndex']], question=row['sQuestion'].strip(), answer=answer,
            original_solutions=row['lSolutions'], equations=row['lEquations'])
    questions = list(unique.values())
    if len(questions) != 580:
        raise ValueError('Expected 580 unique MultiArith questions after whitespace normalization.')
    random.Random(42).shuffle(questions)
    return questions[:290], questions[290:]


def prepare_selection(source, output):
    source.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    path = source/'MultiArith-source.json'
    if not path.exists():
        data = reasoning.download(SOURCE_URL)
        split_questions(data)  # Validate before caching or spending on TTS.
        path.write_bytes(data)
    train, test = split_questions(path.read_bytes())
    reasoning.freeze(output/'selection.json', dict(dataset='MultiArith', source_url=SOURCE_URL,
        source_revision=SOURCE_REVISION, source_sha256=SOURCE_SHA256,
        source_rows=600, unique_questions=580, seed=42,
        rule='Keep first row per whitespace-normalized question; seed42 shuffle; first 290 collection, last 290 evaluation. No outcome/classifier filtering.',
        collection=train, evaluation=test, prompt=PROMPT, collection_prompt=PROMPT,
        layer=31, group='answer', strength=1, primary_metric='flexible-numeric',
        secondary_metrics=['strict-match', 'flexible-extract', 'format-compliant']))
    reasoning.bank_file(source/'questions.jsonl', train+test)
    return train, test


def prepare(source, output):
    train, test = prepare_selection(source, output)
    questions = train+test
    reasoning.generate_audio(source, questions, ['neutral', 'angry'])
    records = reasoning.source_records(source, {q['question_id']: q for q in questions}, ['neutral', 'angry'])
    bank = {q['question_id']: q for q in test}
    if {q['question_id'] for q in train} & set(bank):
        raise ValueError('Collection and evaluation overlap.')
    return dict(task='multiarith', prompt=PROMPT, collection_prompt=PROMPT,
        vector=str((output/'style-vectors.pt').absolute()), bank=bank,
        source_url=SOURCE_URL, source_sha256=SOURCE_SHA256,
        extraction=[dict(question_id=q['question_id'], neutral=records[q['question_id'], 'neutral'],
                         angry=records[q['question_id'], 'angry']) for q in train],
        recordings=reasoning.tasks_for(records, bank),
        seed=42, max_new_tokens=512, steps=128, block_length=32, temperature=0., decoder='fast_dllm',
        layer=31, group='answer', strength=1, primary_metric='flexible-numeric',
        audio_encoding='Native up to 30s; full unchanged longer audio uses 30s chunked Whisper and global adapters',
        selection='580 unique questions; fixed seed42 290/290 split before outcomes; no classifier/outcome filtering',
        scoring_note='Installed upstream GSM8K answer filters/metric on MultiArith references; numeric equivalence additional. Audio-adapted inference, not an official MultiArith harness task.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['all', 'prepare', 'summary'], default='all')
    parser.add_argument('--source', type=Path, default=Path('outputs/azure-multiarith580'))
    parser.add_argument('--output', type=Path, default=Path('outputs/multiarith580-angry-mean'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        reasoning.report(args.output)
        return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == 'prepare':
            train, test = prepare_selection(args.source, args.output)
            print(f'Frozen {len(train)} collection / {len(test)} evaluation questions; no TTS or inference run.')
            return
        # Check the scorer before TTS. The prompt uses the same #### format for collection and test.
        reasoning.scorer_for(dict(task='multiarith'), args.output)
        plan = prepare(args.source, args.output)
        reasoning.freeze(args.output/'run_config.json', plan)
        print('290 collection / 290 test questions; own MultiArith mean; L31 answer x1; seed42.', flush=True)
        print('1,160 Azure clips; up to 580 collection + 870 evaluation generations.', flush=True)
        print(plan['scoring_note'], flush=True)
        for kind in ('reasoning-transfer', 'mean-steering', 'strength-sweep', 'seven-object'):
            subprocess.run([sys.executable, '-u', 'wait_for_experiments.py', '--kind', kind], check=True)
        Path('outputs').mkdir(exist_ok=True)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True)
                time.sleep(30)
            reasoning.evaluate(plan, args.output)
        reasoning.report(args.output)


if __name__ == '__main__':
    main()
