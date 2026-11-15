"""DIFFA-2 delivery-only evaluation: 200 Web of Lies and 200 Tracking Shuffled Objects questions."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

from rescore_audio_steering import HarnessTaskScorer, download, paired
from run_bbh_logical200 import REVISION
from run_reasoning_transfer import digest, freeze, generate_audio, source_records
from steer import read_rows

STYLES = ('neutral', 'angry', 'happy')
TASKS = ('web_of_lies', 'tracking_shuffled_objects_three_objects')
CHECKSUMS = dict(web_of_lies='3e60323f8514a107fdb4edf35f8832a9171c5a09c0eaa47823559de905e8ab8b',
    tracking_shuffled_objects_three_objects='546cc7bf10c6a01a6a4c17ae28a4966485f21f29fd5d111f37858ec78e354f6b')
PROMPTS = dict(
    web_of_lies=("Evaluate the truth-telling problem in the audio. Let's think step by step. "
        'End with exactly one plain-text line in this format: The answer is <yes-or-no>. '
        'Replace <yes-or-no> with the correct answer yes or no.'),
    tracking_shuffled_objects_three_objects=("Track the object swaps described in the audio and answer its question. "
        "The audio includes choices A, B, C. Let's think step by step. "
        'End with exactly one plain-text line in this format: The answer is (<letter>). '
        'Replace <letter> with the correct option A, B, or C.'))


def questions(data, task, count=200):
    if hashlib.sha256(data).hexdigest() != CHECKSUMS[task]:
        raise ValueError('Official BBH source checksum differs.')
    examples = list(enumerate(json.loads(data)['examples']))
    random.Random(42).shuffle(examples)
    if not 1 <= count <= len(examples):
        raise ValueError('Requested more questions than the official source contains.')
    selected = []
    for i, example in examples[:count]:
        spoken = example['input']
        if task == 'web_of_lies':
            if example['target'].lower() not in ('yes', 'no'):
                raise ValueError('Expected yes/no target.')
        else:
            stem, options = spoken.rsplit('\nOptions:\n', 1)
            choices = [m.groups() for line in options.splitlines() if (m := re.fullmatch(r'\(([A-C])\) (.+)', line))]
            if [c[0] for c in choices] != list('ABC') or example['target'] not in ('(A)', '(B)', '(C)'):
                raise ValueError('Unexpected tracking-task options.')
            spoken = stem+'\nOptions:\n'+'\n'.join(f'Option {letter}: {text}' for letter, text in choices)
        selected.append(dict(question_id=f'bbh_{task}_{i:03d}', source_index=i,
            original_input=example['input'], original_target=example['target'],
            question=spoken, answer=example['target']))
    return selected


def prepare(args, task):
    source = args.source/task; source.mkdir(parents=True, exist_ok=True)
    original = source/'bbh-source.json'
    if not original.exists():
        original.write_bytes(download(f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{REVISION}/bbh/{task}.json'))
    selected = questions(original.read_bytes(), task, args.num_examples)
    generate_audio(source, selected, list(STYLES))
    bank = {q['question_id']: q for q in selected}
    clips = source_records(source, bank, STYLES)
    recordings = [dict(clips[q, s], condition=s) for q in bank for s in STYLES]
    return dict(task=task, harness_task='bbh_cot_zeroshot_'+task, bank=bank, recordings=recordings,
        prompt=PROMPTS[task], seed=42, max_new_tokens=512, steps=128, block_length=32,
        temperature=0., decoder='fast_dllm', revision=REVISION, source_sha256=CHECKSUMS[task],
        primary_metric='flexible-extract', interventions=[],
        selection='Fixed seed42 shuffle before any model outcome; same questions and voice in all styles; no classifier filtering',
        audio_encoding='Native up to 30s; longer unchanged audio uses 30s chunked Whisper with global adapters')


def scorer_for(plan, output):
    return HarnessTaskScorer(plan['harness_task'], output/'harness/upstream')


def score(scorer, plan, rows):
    docs = [dict(input=plan['bank'][r['question_id']]['original_input'],
                 target=plan['bank'][r['question_id']]['original_target']) for r in rows]
    return scorer.score([r['response'] for r in rows], docs)


def report(output):
    config = output/'run_config.json'
    if not config.exists():
        print(f'{output.name}: not started.', flush=True); return
    plan = json.loads(config.read_text()); scorer = scorer_for(plan, output)
    latest = {(r['question_id'], r['voice'], r['style']): r for r in read_rows(output/'results.jsonl')}
    rows = list(latest.values()); values = score(scorer, plan, rows)
    rescored = [dict(row, harness={name: entries[i] for name, entries in values.items()}) for i, row in enumerate(rows)]
    summary = dict(task=plan['task'], primary_metric='flexible-extract', scorer=scorer.provenance, accuracy=[], paired=[])
    print(f'\n{plan["task"]}: {len(rows)}/{len(plan["recordings"])} answers; no steering.', flush=True)
    for metric in values:
        groups = {s: {(r['question_id'], r['voice']): r['harness'][metric]['correct'] for r in rescored if r['style'] == s}
                  for s in STYLES}
        common = sorted(set.intersection(*(set(v) for v in groups.values())))
        print(f'\n{metric}: {len(common)} questions completed in all three deliveries.')
        for style, answers in groups.items():
            correct = sum(answers[k] for k in common); n = len(common)
            summary['accuracy'].append(dict(metric=metric, condition=style, n=n, correct=correct))
            print(f'  {style}: {correct}/{n}' + (f' ({correct/n:.1%})' if n else ''))
        for style in STYLES[1:]:
            if not common:
                continue
            result = paired([(groups['neutral'][k], groups[style][k]) for k in common])
            summary['paired'].append(dict(metric=metric, condition=style, **result))
            lo, hi = result['ci95_gain_pp']
            print(f'  {style} vs neutral: gain={result["gain_pp"]:+.2f} pp; W/L={result["wins"]}/{result["losses"]}; '
                  f'p={result["mcnemar_exact_p"]:.4f}; CI=[{lo:+.1f}, {hi:+.1f}]')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rescored))
    print('Installed upstream task scoring on a spoken adaptation; p-values/intervals unadjusted.', flush=True)


def run(plan, output):
    done = {(r['question_id'], r['voice'], r['style']) for r in read_rows(output/'results.jsonl', repair=True)}
    pending = [r for r in plan['recordings'] if (r['question_id'], r['voice'], r['style']) not in done]
    if not pending:
        return
    scorer = scorer_for(plan, output)
    cache = Path(os.environ.get('HF_HOME') or next((str(p) for p in
        (Path('/workspace/.hf_home'), Path('/workspace/hf-cache')) if (p/'hub').is_dir()), 'outputs/hf-diffa')).absolute()
    os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache/'hub'), HUGGINGFACE_HUB_CACHE=str(cache/'hub'))
    os.environ.pop('TRANSFORMERS_CACHE', None)
    import numpy as np
    import soundfile as sf
    import torch
    from extract_vectors import gen_chunked, load_model
    model, tokenizer, features = load_model()
    for i, task in enumerate(pending, 1):
        if digest(task['audio_file']) != task['audio_sha256']:
            raise ValueError('Audio changed after preparation.')
        audio, sr = sf.read(task['audio_file'], dtype='float32')
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        print(f'[{i}/{len(pending)}] {task["question_id"]} {task["style"]}', flush=True)
        started = time.monotonic()
        response, encoding = gen_chunked(audio, sr, model, tokenizer, features, plan['prompt'], 512, 30)
        row = dict(question_id=task['question_id'], voice=task['voice'], style=task['style'], condition=task['style'],
            reference_answer=task['answer'], response=response, prompt=plan['prompt'], seed=42,
            max_new_tokens=512, decoder='fast_dllm', audio_file=task['audio_file'], audio_sha256=task['audio_sha256'],
            audio_encoding=encoding, generation_seconds=time.monotonic()-started)
        values = score(scorer, plan, [row]); row['harness'] = {name: results[0] for name, results in values.items()}
        with (output/'results.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(f'  {row["harness"]}\n{response}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['all', *TASKS], default='all')
    p.add_argument('--mode', choices=['all', 'evaluate', 'summary'], default='all')
    p.add_argument('--source', type=Path, default=Path('outputs/azure-bbh-reasoning200'))
    p.add_argument('--output', type=Path, default=Path('outputs/bbh-reasoning200-diffa2'))
    p.add_argument('--num-examples', type=int, default=200)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if not 1 <= args.num_examples <= 250:
        p.error('--num-examples must be between 1 and 250')
    if args.mode == 'summary':
        for task in TASKS if args.task == 'all' else [args.task]:
            report(args.output/task)
        return
    if args.task == 'all':
        with (args.output/'queue.lock').open('w') as own:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for task in TASKS:
                print('\nSTAGE:', task, flush=True)
                (args.output/'status.json').write_text(json.dumps(dict(task=task, status='running'))+'\n')
                subprocess.run([sys.executable, '-u', str(Path(__file__).absolute()), *sys.argv[1:], '--task', task], check=True)
            (args.output/'status.json').write_text(json.dumps(dict(task='finished', status='complete'))+'\n')
        return
    output = args.output/args.task; output.mkdir(parents=True, exist_ok=True)
    with (output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = output/'run_config.json'
        if args.mode == 'all':
            # Verify final-answer formatting against the actual upstream filters before spending on TTS.
            scorer = HarnessTaskScorer('bbh_cot_zeroshot_'+args.task, output/'harness/upstream')
            if args.task == 'web_of_lies':
                responses = ['The answer is yes.', 'The answer is no.']; docs = [dict(input='Truth?', target=t) for t in ('Yes', 'No')]
            else:
                responses = [f'The answer is ({c}).' for c in 'ABC']
                docs = [dict(input='Which?\nOptions:\n(A) red\n(B) blue\n(C) green', target=f'({c})') for c in 'ABC']
            if not all(r['correct'] for pipeline in scorer.score(responses, docs).values() for r in pipeline):
                raise ValueError('Requested answer format does not match the upstream scorer.')
            plan = prepare(args, args.task); freeze(config, plan)
        else:
            plan = json.loads(config.read_text())
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            run(plan, output)
        report(output)


if __name__ == '__main__':
    main()
