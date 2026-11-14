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
PROMPT = ('A logical deduction task which requires deducing the order of a sequence of objects. '
          'Solve the problem in the audio, including its three choices A, B, C. '
          "Let's think step by step. End with exactly one plain-text line in this format: "
          'The answer is (<letter>). Replace <letter> with the correct option A, B, or C.')


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


def recordings(source, bank, allow_long=False):
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
        if any(row['seconds'] <= 0 or (not allow_long and row['seconds'] > 30) for row in group.values()):
            excluded.append(qid)
        else:
            tasks.extend(dict(group[s], condition=s) for s in STYLES)
    if not tasks:
        raise ValueError('No matched four-style questions fit 30 seconds. For unchanged long audio, use --chunked-audio.')
    rule = ('Every style must be nonempty; full audio encoded in chunks, no cropping or correctness filtering'
            if allow_long else 'Every style must fit 30 seconds; no cropping or correctness filtering')
    return tasks, dict(rule=rule,
                       excluded_questions=excluded, durations=durations)


def summarize(args):
    from score_bbh_harness import report_saved
    report_saved(args.output, args.source)


def evaluate(args):
    bank = {q['question_id']: q for q in read_rows(args.source/'questions.jsonl')}
    chunked = getattr(args, 'chunked_audio', False)
    tasks, eligibility = recordings(args.source, bank, allow_long=chunked)
    if getattr(args, 'smoke_test', False):
        ids = list(dict.fromkeys(t['question_id'] for t in tasks))[:2]
        tasks = [t for t in tasks if t['question_id'] in ids]
    config = dict(recordings=tasks, eligibility=eligibility, seed=42, prompt=PROMPT,
                  max_new_tokens=512, steps=128, block_length=32, temperature=0., decoder='fast_dllm',
                  interventions=[], source_sha256=SOURCE_SHA256)
    if chunked:
        config['audio_encoding'] = dict(mode='chunked-whisper-global-adapters-v1',
            chunk_seconds=args.chunk_seconds, smoke_test=args.smoke_test)
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
            if chunked:
                run(pending, bank, args.output, chunked=True, chunk_seconds=args.chunk_seconds,
                    smoke_test=args.smoke_test)
            else:
                run(pending, bank, args.output)
    summarize(args)


def run(pending, bank, output, chunked=False, chunk_seconds=30, smoke_test=False):
    cache = Path(os.environ.get('HF_HOME') or next((str(p) for p in
        (Path('/workspace/.hf_home'), Path('/workspace/hf-cache')) if (p/'hub').is_dir()), 'outputs/hf-diffa')).absolute()
    os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache/'hub'), HUGGINGFACE_HUB_CACHE=str(cache/'hub'))
    os.environ.pop('TRANSFORMERS_CACHE', None)
    import numpy as np
    import soundfile as sf
    import torch
    from extract_vectors import gen, gen_chunked, load_model
    from score_bbh_harness import HarnessScorer
    harness = HarnessScorer(output/'harness/upstream')
    model, tokenizer, features = load_model()
    def reset_seed():
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    if smoke_test:
        audio, sr = sf.read(pending[0]['audio_file'], dtype='float32')
        short = audio[:int(20*sr)]
        reset_seed()
        original = gen(short, sr, model, tokenizer, features, PROMPT, 512)
        reset_seed()
        adapted, _ = gen_chunked(short, sr, model, tokenizer, features, PROMPT, 512)
        if original != adapted:
            raise ValueError('Short-audio equivalence check failed; no BBH answers saved.')
        checks = dict(short_audio_identical=True, short_response=original, boundary_checks=[])
        long_task = max(pending, key=lambda task: task['seconds'])
        audio, sr = sf.read(long_task['audio_file'], dtype='float32')
        for boundary in (20, 30):
            reset_seed()
            response, encoding = gen_chunked(audio, sr, model, tokenizer, features, PROMPT, 512, boundary)
            checks['boundary_checks'].append(dict(question_id=long_task['question_id'], style=long_task['style'],
                response=response, encoding=encoding, custom_scores=score(response, bank[long_task['question_id']])))
        (output/'chunking-preflight.json').write_text(json.dumps(checks, indent=2)+'\n')
        print('Short-audio check: identical response. 20s/30s boundary diagnostics saved to chunking-preflight.json.', flush=True)
    for index, task in enumerate(pending, 1):
        path = Path(task['audio_file'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != task['audio_sha256']:
            raise ValueError('BBH audio changed.')
        audio, sr = sf.read(path, dtype='float32')
        reset_seed()
        print(f'[{index}/{len(pending)}] {task["question_id"]} {task["style"]}', flush=True)
        started = time.monotonic()
        if chunked:
            response, encoding = gen_chunked(audio, sr, model, tokenizer, features, PROMPT, 512, chunk_seconds)
            print('Audio encoding:', encoding, flush=True)
        else:
            response = gen(audio, sr, model, tokenizer, features, PROMPT, 512)
        row = dict(question_id=task['question_id'], voice=task['voice'], style=task['style'],
            condition=task['style'], audio_path=task['audio_path'], seed=42, prompt=PROMPT,
            max_new_tokens=512, decoder='fast_dllm', response=response,
            generation_seconds=time.monotonic()-started, **score(response, bank[task['question_id']]))
        question = bank[task['question_id']]
        values = harness.score([response], [dict(input=question['original_input'], target=question['original_target'])])
        row['harness'] = {name: results[0] for name, results in values.items()}
        if chunked:
            row['audio_encoding'] = encoding
        with (output/'results.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        status = '; '.join(f'{name}={value["correct"]}' for name, value in row['harness'].items())
        print(f'  {status}\n{response}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['all', 'prepare', 'generate', 'evaluate', 'summary'], default='all')
    p.add_argument('--source', type=Path, default=Path('outputs/azure-bbh-logical200'))
    p.add_argument('--output', type=Path)
    p.add_argument('--num-examples', type=int, default=200)
    p.add_argument('--chunked-audio', action='store_true', help='Encode full unchanged long audio; apply adapters once after feature concatenation')
    p.add_argument('--chunk-seconds', type=float, default=30, help='Whisper window, at most 30 seconds')
    p.add_argument('--smoke-test', action='store_true', help='With --chunked-audio: two questions/four styles plus short-audio and boundary checks')
    args = p.parse_args()
    if not 0 < args.chunk_seconds <= 30:
        p.error('--chunk-seconds must be >0 and <=30')
    if args.smoke_test and not args.chunked_audio:
        p.error('--smoke-test requires --chunked-audio')
    if args.output is None:
        name = 'outputs/bbh-logical200-diffa2-harness'
        if args.chunked_audio:
            name += '-chunked-smoke' if args.smoke_test else '-chunked'
        args.output = Path(name)
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
