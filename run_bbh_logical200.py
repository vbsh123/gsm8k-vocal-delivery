"""200 BBH three-object logical deduction questions, four Azure deliveries, DIFFA-2 accuracy only."""

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
from urllib.request import urlopen
import wave

from evaluate_arc_audio import report, score
from steer import read_rows

REVISION = '9ee07bd481feebf959a6b59d61ea57bdcf30964d'
SOURCE_URL = f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{REVISION}/bbh/logical_deduction_three_objects.json'
SOURCE_SHA256 = 'e590fae72e3a6041fbd23bd8ff6b74baf44ca563f6ba40a041c55e7b867c1681'
STYLES = ['neutral', 'angry', 'sad', 'happy']
PROMPT = ('Solve the logical deduction problem in the audio. The audio includes three choices labeled A, B, C. '
          'Give concise reasoning. End with one plain-text line in exactly this format: Final answer: <letter>. '
          'Give only one letter from A to C on that line.')


def select_questions(data, count=200):
    rows = list(enumerate(json.loads(data)['examples']))
    random.Random(42).shuffle(rows)
    if not 1 <= count <= len(rows):
        raise ValueError('BBH three-object task has only 250 examples.')
    selected = []
    for index, row in rows[:count]:
        stem, options = row['input'].rsplit('\nOptions:\n', 1)
        choices = [dict(label=match[1], text=match[2]) for line in options.splitlines()
                   if (match := re.fullmatch(r'\(([A-C])\) (.+)', line))]
        if [c['label'] for c in choices] != list('ABC') or row['target'] not in ('(A)', '(B)', '(C)'):
            raise ValueError('Unexpected BBH choice format.')
        spoken = stem+'\nOptions:\n'+'\n'.join(f'Option {c["label"]}: {c["text"]}' for c in choices)
        selected.append(dict(question_id=f'bbh_logical_three_{index:03d}', source_index=index,
            question=spoken, original_input=row['input'], original_target=row['target'],
            choices=choices, answer=row['target'][1], source_task='logical_deduction_three_objects'))
    return selected


def generate(args):
    args.source.mkdir(parents=True, exist_ok=True)
    path = args.source/'logical_deduction_three_objects-source.json'
    if not path.exists():
        with urlopen(SOURCE_URL, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
            raise ValueError('BBH source checksum changed.')
        path.write_bytes(data)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('Cached BBH source checksum differs.')
    questions = select_questions(data, args.num_examples)
    text = ''.join(json.dumps(q, ensure_ascii=False)+'\n' for q in questions)
    bank = args.source/'questions.jsonl'
    if bank.exists() and bank.read_text() != text:
        raise ValueError('Selected questions changed; use a new source folder.')
    bank.write_text(text)
    (args.source/'selection.json').write_text(json.dumps(dict(
        source_url=SOURCE_URL, source_sha256=SOURCE_SHA256, revision=REVISION, seed=42,
        task='logical_deduction_three_objects', selected_indices=[q['source_index'] for q in questions],
        adaptation='Original wording; (A)/(B)/(C) option markers spoken as Option A/B/C',
        styles=STYLES, voice='en-US-JennyNeural', intensity=1.5), indent=2)+'\n')
    print(f'{len(questions)} fixed three-object logical-deduction questions × four deliveries; NO steering or extraction.', flush=True)
    command = [sys.executable, '-u', 'cloud_tts_smoke.py', '--provider', 'azure',
        '--azure-voice', 'en-US-JennyNeural', '--azure-intensity', '1.5', '--input-jsonl', str(bank),
        '--num-examples', str(len(questions)), '--styles', *STYLES, '--output', str(args.source),
        '--resume', '--retry-passes', '5']
    if args.mode == 'prepare':
        command.append('--prepare-only')
    subprocess.run(command, check=True)


def recordings(source, bank):
    groups = {}; durations = []
    for row in read_rows(source/'plan.jsonl'):
        if row['provider'] != 'azure' or row['style'] not in STYLES:
            continue
        q = bank[row['question_id']]
        if row['question'] != q['question'] or row['answer'] != q['answer']:
            raise ValueError('Audio plan differs from BBH questions.')
        path = (source/row['audio_path']).absolute()
        with wave.open(str(path), 'rb') as audio:
            seconds = audio.getnframes()/audio.getframerate()
        group = groups.setdefault((row['question_id'], row['voice']), {})
        if row['style'] in group:
            raise ValueError('Duplicate BBH delivery.')
        group[row['style']] = dict(row, audio_file=str(path), seconds=seconds,
                                  audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        durations.append(dict(question_id=row['question_id'], style=row['style'], seconds=seconds))
    tasks = []; excluded = []
    for (qid, _), group in groups.items():
        if set(group) != set(STYLES):
            raise ValueError('Finish generating all four styles before evaluation.')
        if any(not 0 < row['seconds'] <= 30 for row in group.values()):
            excluded.append(qid)
        else:
            tasks.extend(dict(group[s], condition=s) for s in STYLES)
    if not tasks:
        raise ValueError('No matched four-style questions fit 30 seconds.')
    return tasks, dict(rule='Every style must fit 30 seconds; no cropping or correctness filtering',
                       excluded_questions=excluded, durations=durations)


def summarize(args):
    from score_bbh_harness import report_saved
    report_saved(args.output, args.source)


def evaluate(args):
    bank = {q['question_id']: q for q in read_rows(args.source/'questions.jsonl')}
    tasks, eligibility = recordings(args.source, bank)
    config = dict(recordings=tasks, eligibility=eligibility, seed=42, prompt=PROMPT,
                  max_new_tokens=512, steps=128, block_length=32, temperature=0., decoder='fast_dllm',
                  interventions=[], source_sha256=SOURCE_SHA256)
    path = args.output/'run_config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Settings or audio changed; use a separate result folder.')
    path.write_text(json.dumps(config, indent=2)+'\n')
    done = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(args.output/'results.jsonl', repair=True)}
    pending = [t for t in tasks if (t['question_id'], t['voice'], t['condition']) not in done]
    print(f'{len(tasks)//4} matched questions; {len(pending)}/{len(tasks)} answers pending. '
          f'{len(eligibility["excluded_questions"])} overlength/empty questions excluded.', flush=True)
    if pending:
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True)
                time.sleep(30)
            run(pending, bank, args.output)
    summarize(args)


def run(pending, bank, output):
    cache = Path(os.environ.get('HF_HOME') or next((str(p) for p in
        (Path('/workspace/.hf_home'), Path('/workspace/hf-cache')) if (p/'hub').is_dir()), 'outputs/hf-diffa')).absolute()
    os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache/'hub'), HUGGINGFACE_HUB_CACHE=str(cache/'hub'))
    os.environ.pop('TRANSFORMERS_CACHE', None)
    import numpy as np
    import soundfile as sf
    import torch
    from extract_vectors import gen, load_model
    model, tokenizer, features = load_model()
    for index, task in enumerate(pending, 1):
        path = Path(task['audio_file'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != task['audio_sha256']:
            raise ValueError('BBH audio changed.')
        audio, sr = sf.read(path, dtype='float32')
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        print(f'[{index}/{len(pending)}] {task["question_id"]} {task["style"]}', flush=True)
        started = time.monotonic()
        response = gen(audio, sr, model, tokenizer, features, PROMPT, 512)
        row = dict(question_id=task['question_id'], voice=task['voice'], style=task['style'],
            condition=task['style'], audio_path=task['audio_path'], seed=42, prompt=PROMPT,
            max_new_tokens=512, decoder='fast_dllm', response=response,
            generation_seconds=time.monotonic()-started, **score(response, bank[task['question_id']]))
        with (output/'results.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(f'  custom_answer_correct={row["answer_correct"]}; custom_strict_correct={row["correct"]}\n{response}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['all', 'prepare', 'generate', 'evaluate', 'summary'], default='all')
    p.add_argument('--source', type=Path, default=Path('outputs/azure-bbh-logical200'))
    p.add_argument('--output', type=Path, default=Path('outputs/bbh-logical200-diffa2'))
    p.add_argument('--num-examples', type=int, default=200)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        summarize(args)
        return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode != 'prepare':
            subprocess.run([sys.executable, '-u', 'wait_for_experiments.py', '--kind', 'svamp'], check=True)
        if args.mode in ('all', 'prepare', 'generate'):
            generate(args)
        if args.mode in ('all', 'evaluate'):
            evaluate(args)


if __name__ == '__main__':
    main()
