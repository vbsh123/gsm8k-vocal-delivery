"""Azure emotional spoken IFEval: audio instructions, text responses, official checkers."""

import argparse
from collections import defaultdict
import fcntl
import hashlib
from itertools import combinations
import json
from math import comb
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from urllib.request import urlopen
import wave

from steer import read_rows

REVISION = 'e6890f85757dd84e27ca6df2dd30651dafad28e0'
OFFICIAL = Path('external/ifeval')
STYLES = ['neutral', 'angry', 'happy', 'sad']
PROMPT = 'Follow the instructions in the audio.'


def setup_official():
    package = OFFICIAL / 'instruction_following_eval'
    for name in ['evaluation_lib.py', 'instructions.py', 'instructions_registry.py',
                 'instructions_util.py', 'requirements.txt', 'data/input_data.jsonl']:
        target = package / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://raw.githubusercontent.com/google-research/google-research/{REVISION}/instruction_following_eval/{name}'
            with urlopen(url, timeout=120) as response:
                data = response.read()
            temporary = target.with_suffix('.tmp')
            temporary.write_bytes(data)
            temporary.replace(target)
    return package


def prepare(args):
    package = setup_official()
    examples = read_rows(package / 'data/input_data.jsonl')
    if args.num_examples:
        examples = examples[:args.num_examples]
    questions = [dict(question_id=f'ifeval_{e["key"]}', question=e['prompt'], **e) for e in examples]
    args.source.mkdir(parents=True, exist_ok=True)
    bank = args.source / 'questions.jsonl'
    text = ''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in questions)
    if bank.exists() and bank.read_text() != text:
        raise ValueError('Different selected prompts; use a separate source folder.')
    bank.write_text(text)
    manifest = dict(benchmark='Our spoken adaptation of original IFEval', revision=REVISION,
                    hashes={str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in package.rglob('*') if p.is_file() and p.suffix in ('.py', '.jsonl')},
                    selection='Original source order; no outcome or classifier filtering',
                    questions=len(questions), styles=STYLES, voice='en-US-JennyNeural', intensity=1.5)
    (args.source / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    command = [sys.executable, '-u', 'cloud_tts_smoke.py', '--provider', 'azure',
               '--input-jsonl', str(bank), '--num-examples', str(len(questions)),
               '--styles', *STYLES, '--output', str(args.source), '--resume', '--retry-passes', '5']
    if args.mode == 'prepare':
        command.append('--prepare-only')
    print(f'{len(questions)} original IFEval prompts × four deliveries; unchanged prompt wording.', flush=True)
    subprocess.run(command, check=True)


def recordings(source):
    grouped = defaultdict(dict)
    durations = []
    for task in read_rows(source / 'plan.jsonl'):
        if task['provider'] != 'azure' or task['style'] not in STYLES:
            continue
        file = source / task['audio_path']
        with wave.open(str(file), 'rb') as audio:
            seconds = audio.getnframes() / audio.getframerate()
        grouped[task['question_id'], task['voice']][task['style']] = task
        durations.append(dict(question_id=task['question_id'], style=task['style'], seconds=seconds))
    excluded = {r['question_id'] for r in durations if not 0 < r['seconds'] <= 30}
    eligible = []
    for (question, _), styles in grouped.items():
        if set(styles) != set(STYLES):
            raise ValueError(f'Missing delivery for {question}; finish generation first.')
        if question not in excluded:
            eligible.extend(styles[s] for s in STYLES)
    (source / 'eligibility.json').write_text(json.dumps(dict(
        rule='All four recordings must be nonempty and at most 30 seconds; no cropping',
        eligible_questions=len(eligible) // 4, excluded_questions=sorted(excluded), durations=durations), indent=2) + '\n')
    return eligible


def official_scorer():
    sys.path.insert(0, str(OFFICIAL.absolute()))
    from langdetect import DetectorFactory
    from instruction_following_eval import evaluation_lib
    DetectorFactory.seed = 42
    return evaluation_lib


def score(example, response, evaluator):
    inp = evaluator.InputExample(**{k: example[k] for k in ('key', 'prompt', 'instruction_id_list', 'kwargs')})
    result = {}
    for mode in ('strict', 'loose'):
        random.seed(42)
        checker = getattr(evaluator, 'test_instruction_following_' + mode)
        checked = checker(inp, {inp.prompt: response})
        result[mode] = dict(prompt=bool(checked.follow_all_instructions),
                            instructions=[bool(x) for x in checked.follow_instruction_list])
    return result


def report(args):
    import numpy as np
    examples = {e['question_id']: e for e in read_rows(args.source / 'questions.jsonl')}
    evaluator = official_scorer()
    latest = {(r['question_id'], r['voice'], r['style']): r for r in read_rows(args.output / 'results.jsonl')}
    groups = {style: {} for style in STYLES}
    scored = []
    for (qid, voice, style), row in latest.items():
        metrics = score(examples[qid], row['response'], evaluator)
        groups[style][qid, voice] = metrics
        scored.append(dict(row, official_scores=metrics))
    (args.output / 'official-scored.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in scored))
    print('\nOfficial IFEval checks — our spoken adaptation; completed answers only.')
    totals = []
    for style in STYLES:
        values = list(groups[style].values())
        if not values:
            continue
        total = dict(style=style, prompts=len(values))
        for mode in ('strict', 'loose'):
            bits = [b for v in values for b in v[mode]['instructions']]
            total[mode + '_prompt'] = sum(v[mode]['prompt'] for v in values) / len(values)
            total[mode + '_instruction'] = sum(bits) / len(bits)
        totals.append(total)
        print(f'{style:8} N={len(values):3}  strict prompt={total["strict_prompt"]:.1%}'
              f' instruction={total["strict_instruction"]:.1%}  loose prompt={total["loose_prompt"]:.1%}'
              f' instruction={total["loose_instruction"]:.1%}')
    comparisons = []
    for mode in ('strict', 'loose'):
        print(f'\n{mode} prompt-level paired comparisons:')
        print('Reference Test      N    Ref     Test  Gain pp  Wins Losses  Exact p    95% CI pp')
        for a, b in combinations(STYLES, 2):
            keys = sorted(groups[a].keys() & groups[b].keys())
            n = len(keys)
            if not n:
                continue
            left = [groups[a][k][mode]['prompt'] for k in keys]
            right = [groups[b][k][mode]['prompt'] for k in keys]
            w = sum(y and not x for x, y in zip(left, right))
            l = sum(x and not y for x, y in zip(left, right))
            d = w + l
            p = min(1., 2 * sum(comb(d, i) for i in range(min(w, l) + 1)) / 2 ** d)
            draws = np.random.default_rng(42).multinomial(n, [w/n, l/n, (n-d)/n], size=200000)
            ci = np.percentile(100 * (draws[:, 0] - draws[:, 1]) / n, [2.5, 97.5]).tolist()
            comparisons.append(dict(mode=mode, reference=a, test=b, pairs=n, ref_correct=sum(left),
                                    test_correct=sum(right), wins=w, losses=l, gain_pp=100*(w-l)/n,
                                    mcnemar_exact_p=p, ci95_gain_pp=ci))
            print(f'{a:9} {b:7} {n:3} {sum(left)/n:7.1%} {sum(right)/n:7.1%}'
                  f' {100*(w-l)/n:+7.2f} {w:5} {l:6} {p:8.4f} [{ci[0]:+.1f}, {ci[1]:+.1f}]')
    print('Intervals and p-values are unadjusted. No semantic/factual correctness judgment.')
    near_budget = {s: sum(r.get('response_tokens', 0) >= r.get('max_new_tokens', 1024)-8
                         for r in latest.values() if r['style'] == s) for s in STYLES}
    print('Responses within eight tokens of the output budget (possible truncation):', near_budget)
    (args.output / 'summary.json').write_text(json.dumps(dict(revision=REVISION, totals=totals,
        paired_prompt_comparisons=comparisons, responses_near_token_budget=near_budget), indent=2) + '\n')


def evaluate(args):
    tasks = recordings(args.source)
    if not tasks:
        raise ValueError('No questions fit the matched four-style 30-second cohort.')
    evaluator = official_scorer()
    for example in read_rows(args.source / 'questions.jsonl'):
        score(example, 'Preflight check.', evaluator)
    config = dict(seed=42, prompt=PROMPT, max_new_tokens=args.max_new_tokens,
                  decoder='fast_dllm', steps=128, block_length=32, temperature=0., revision=REVISION,
                  recordings=[dict(t, audio_sha256=hashlib.sha256((args.source/t['audio_path']).read_bytes()).hexdigest())
                              for t in tasks])
    path = args.output / 'run_config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Settings or audio changed; use a separate output folder.')
    path.write_text(json.dumps(config, indent=2) + '\n')
    done = {r['audio_path'] for r in read_rows(args.output / 'results.jsonl', repair=True)}
    pending = [t for t in tasks if t['audio_path'] not in done]
    print(f'{len(tasks)//4} matched questions; {len(pending)}/{len(tasks)} generations pending; seed42.', flush=True)
    if not pending:
        report(args)
        return
    with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
        print('Waiting for the shared GPU lock.', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            print('Waiting for active GPU work.', flush=True)
            time.sleep(30)
        cache = Path(os.environ.get('HF_HOME') or next((str(p) for p in
            (Path('/workspace/hf-cache'), Path('/workspace/.hf_home')) if (p/'hub').is_dir()), 'outputs/hf-diffa')).absolute()
        os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache/'hub'), HUGGINGFACE_HUB_CACHE=str(cache/'hub'))
        os.environ.pop('TRANSFORMERS_CACHE', None)
        import soundfile as sf
        import numpy as np
        import torch
        from extract_vectors import gen, load_model
        model, tokenizer, features = load_model()
        for index, task in enumerate(pending, 1):
            audio, sr = sf.read(args.source / task['audio_path'], dtype='float32')
            random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
            print(f'[{index}/{len(pending)}] {task["question_id"]} {task["style"]}', flush=True)
            started = time.monotonic()
            response = gen(audio, sr, model, tokenizer, features, PROMPT, args.max_new_tokens)
            row = dict(task, response=response, seed=42, wrapper=PROMPT, max_new_tokens=args.max_new_tokens,
                       generation_seconds=time.monotonic()-started,
                       response_tokens=len(tokenizer.encode(response, add_special_tokens=False)))
            with (args.output / 'results.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(response, flush=True)
    report(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['all', 'prepare', 'generate', 'evaluate', 'summary'], default='all')
    p.add_argument('--source', type=Path, default=Path('outputs/azure-spoken-ifeval'))
    p.add_argument('--output', type=Path, default=Path('outputs/spoken-ifeval-diffa2'))
    p.add_argument('--num-examples', type=int, default=0, help='First N prompts; 0 selects all 541 original prompts')
    p.add_argument('--max-new-tokens', type=int, default=1024)
    args = p.parse_args()
    if args.num_examples < 0 or args.max_new_tokens < 1:
        p.error('Invalid count or token budget')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        report(args)
        return
    with (args.output / 'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode in ('all', 'prepare', 'generate'):
            prepare(args)
        if args.mode in ('all', 'evaluate'):
            setup_official()
            evaluate(args)


if __name__ == '__main__':
    main()
