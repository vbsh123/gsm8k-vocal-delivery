"""IFBench neutral/angry/happy audio, DIFFA-2 text responses and released official verifiers."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from urllib.request import urlopen

from run_spoken_ifeval import PROMPT, recordings, report, score
from steer import read_rows

REVISION = '488918665d53f867fa908a293769ec6e71e74aec'
UPSTREAM = Path('external/ifbench')
FILES = ['evaluation_lib.py', 'ifbench/__init__.py', 'ifbench/classic_instructions.py',
         'ifbench/instructions.py', 'ifbench/instructions_registry.py', 'ifbench/instructions_util.py',
         'ifbench/data/IFBench_test.jsonl', 'LICENSE']
DATA_SHA256 = 'a2bf7b8e7ed11d39c65b4ec781da8b99663ed0121f97033d70fcecb61f8a9d85'
STYLES = ['neutral', 'angry', 'happy']


def setup_official():
    for name in FILES:
        path = UPSTREAM/name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(f'https://raw.githubusercontent.com/allenai/IFBench/{REVISION}/{name}', timeout=120) as response:
                data = response.read()
            temporary = path.with_suffix('.tmp'); temporary.write_bytes(data); temporary.replace(path)
    data = (UPSTREAM/'ifbench/data/IFBench_test.jsonl').read_bytes()
    if hashlib.sha256(data).hexdigest() != DATA_SHA256:
        raise ValueError('Official IFBench test data checksum differs.')
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def official_scorer():
    sys.path.insert(0, str(UPSTREAM.absolute()))
    from langdetect import DetectorFactory
    DetectorFactory.seed = 42
    spec = importlib.util.spec_from_file_location('ifbench_official_evaluation', UPSTREAM/'evaluation_lib.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare(args, examples):
    selected = examples[:args.num_examples] if args.num_examples else examples
    questions = [dict(example, question_id=f'ifbench_{example["key"]}', question=example['prompt']) for example in selected]
    args.source.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in questions)
    bank = args.source/'questions.jsonl'
    if bank.exists() and bank.read_text() != text:
        raise ValueError('Selected prompts changed; use a separate source folder.')
    bank.write_text(text)
    (args.source/'source_manifest.json').write_text(json.dumps(dict(
        benchmark='Our spoken adaptation of IFBench test', revision=REVISION, data_sha256=DATA_SHA256,
        source_hashes={name: hashlib.sha256((UPSTREAM/name).read_bytes()).hexdigest() for name in FILES},
        selection='Original source order; all 300 test prompts by default, before duration eligibility',
        styles=STYLES, questions=len(questions), voice='en-US-JennyNeural', intensity=1.5), indent=2)+'\n')
    command = [sys.executable, '-u', 'cloud_tts_smoke.py', '--provider', 'azure',
        '--azure-voice', 'en-US-JennyNeural', '--azure-intensity', '1.5', '--input-jsonl', str(bank),
        '--num-examples', str(len(questions)), '--styles', *STYLES, '--output', str(args.source),
        '--resume', '--retry-passes', '5']
    if args.mode == 'prepare':
        command.append('--prepare-only')
    print(f'{len(questions)} IFBench prompts × neutral/angry/happy = {3*len(questions)} clips; no steering.', flush=True)
    subprocess.run(command, check=True)


def summarize(args, evaluator):
    report(args, styles=STYLES, evaluator=evaluator, benchmark='IFBench', revision=REVISION,
           primary_metric='loose_prompt')


def evaluate(args, evaluator):
    tasks = recordings(args.source, STYLES)
    if not tasks:
        raise ValueError('No matched three-style questions fit 30 seconds; see source eligibility.json.')
    for task in tasks:
        task['audio_sha256'] = hashlib.sha256((args.source/task['audio_path']).read_bytes()).hexdigest()
    config = dict(seed=42, prompt=PROMPT, max_new_tokens=args.max_new_tokens, decoder='fast_dllm',
        steps=128, block_length=32, temperature=0., revision=REVISION, primary_metric='loose_prompt',
        verifier_hashes={name: hashlib.sha256((UPSTREAM/name).read_bytes()).hexdigest() for name in FILES},
        recordings=tasks)
    path = args.output/'run_config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Audio, evaluator or settings changed; use a separate result folder.')
    path.write_text(json.dumps(config, indent=2)+'\n')
    complete = {row['audio_path'] for row in read_rows(args.output/'results.jsonl', repair=True)}
    pending = [t for t in tasks if t['audio_path'] not in complete]
    print(f'{len(tasks)//3} matched three-style questions; {len(pending)}/{len(tasks)} answers pending.', flush=True)
    if pending:
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for the shared GPU lock.', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            run(args, pending)
    summarize(args, evaluator)


def run(args, pending):
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
        path = args.source/task['audio_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != task['audio_sha256']:
            raise ValueError('IFBench audio changed.')
        audio, sr = sf.read(path, dtype='float32')
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        print(f'[{index}/{len(pending)}] {task["question_id"]} {task["style"]}', flush=True)
        started = time.monotonic()
        response = gen(audio, sr, model, tokenizer, features, PROMPT, args.max_new_tokens)
        row = dict(question_id=task['question_id'], voice=task['voice'], style=task['style'],
            audio_path=task['audio_path'], response=response, seed=42, wrapper=PROMPT,
            max_new_tokens=args.max_new_tokens, decoder='fast_dllm', generation_seconds=time.monotonic()-started,
            response_tokens=len(tokenizer.encode(response, add_special_tokens=False)))
        with (args.output/'results.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(response, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['all', 'prepare', 'generate', 'evaluate', 'summary'], default='all')
    p.add_argument('--source', type=Path, default=Path('outputs/azure-spoken-ifbench'))
    p.add_argument('--output', type=Path, default=Path('outputs/spoken-ifbench-diffa2'))
    p.add_argument('--num-examples', type=int, default=0, help='0 selects all 300 test prompts')
    p.add_argument('--max-new-tokens', type=int, default=1024)
    args = p.parse_args()
    if args.num_examples < 0 or args.max_new_tokens <= 0:
        p.error('Invalid count or output budget')
    args.output.mkdir(parents=True, exist_ok=True)
    examples = setup_official()
    evaluator = official_scorer()
    if args.mode == 'summary':
        summarize(args, evaluator)
        return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Exercise every requested verifier before spending Azure/GPU time.
        for example in examples[:args.num_examples] if args.num_examples else examples:
            score(example, 'Preflight check.', evaluator)
        if args.mode in ('all', 'prepare', 'generate'):
            prepare(args, examples)
        if args.mode in ('all', 'evaluate'):
            evaluate(args, evaluator)


if __name__ == '__main__':
    main()
