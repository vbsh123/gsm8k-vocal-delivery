"""BBH: collect an angry-minus-neutral L31 mean on 125 questions, test on the other 125."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import wave

from extract_audio_vectors import GROUPS, save_tensor
from rescore_audio_steering import paired
from run_bbh_logical200 import PROMPT, SOURCE_SHA256, select_questions
from steer import install_interventions, read_rows

CONDITION = 'neutral-BBH-L31-answer-a1'
CONDITIONS = ('neutral', 'angry', CONDITION)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def prepare(args):
    original = args.existing_source/'logical_deduction_three_objects-source.json'
    data = original.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('BBH source checksum differs.')
    questions = select_questions(data, 250)
    args.source.mkdir(parents=True, exist_ok=True)
    (args.source/original.name).write_bytes(data)
    (args.source/'questions.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in questions))
    command = [sys.executable, '-u', 'cloud_tts_smoke.py', '--provider', 'azure',
        '--azure-voice', 'en-US-JennyNeural', '--azure-intensity', '1.5',
        '--input-jsonl', str(args.source/'questions.jsonl'), '--num-examples', '250',
        '--styles', 'neutral', 'angry', '--output', str(args.source), '--resume', '--retry-passes', '5']
    subprocess.run(command+['--prepare-only'], check=True)
    previous = {(r['question_id'], r['voice'], r['style']): r
                for r in read_rows(args.existing_source/'plan.jsonl') if r['provider'] == 'azure'}
    for task in read_rows(args.source/'plan.jsonl'):
        old = previous.get((task['question_id'], task['voice'], task['style']))
        if old is None:
            continue
        if any(old[k] != task[k] for k in ('question', 'answer', 'request')):
            raise ValueError('Existing BBH recording uses different wording or TTS settings.')
        source, target = args.existing_source/old['audio_path'], args.source/task['audio_path']
        if source.exists() and not target.exists():
            target.symlink_to(source.absolute())
    print('Reusing existing neutral/angry audio; only the 50 unused questions need new TTS (100 clips).', flush=True)
    if args.mode != 'prepare':
        subprocess.run(command, check=True)


def make_plan(source):
    questions = read_rows(source/'questions.jsonl')
    if len(questions) != 250 or len({q['question_id'] for q in questions}) != 250:
        raise ValueError('Expected all 250 distinct BBH three-object questions.')
    bank = {q['question_id']: q for q in questions}
    groups = {}
    for row in read_rows(source/'plan.jsonl'):
        q = bank[row['question_id']]
        if row['question'] != q['question'] or row['answer'] != q['answer']:
            raise ValueError('Audio wording/reference differs from BBH source.')
        path = (source/row['audio_path']).absolute()
        with wave.open(str(path), 'rb') as audio:
            seconds = audio.getnframes()/audio.getframerate()
        if seconds <= 0:
            raise ValueError('Empty recording.')
        styles = groups.setdefault(row['question_id'], {})
        if row['style'] in styles or row['provider'] != 'azure':
            raise ValueError('Duplicate or non-Azure BBH recording.')
        styles[row['style']] = dict(row, audio_file=str(path), seconds=seconds,
                                   audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if set(groups) != set(bank) or any(set(g) != {'neutral', 'angry'} for g in groups.values()):
        raise ValueError('Finish generating all 250 neutral/angry pairs.')
    # select_questions already uses the established seed42 shuffled source order.
    ids = [q['question_id'] for q in questions]
    return dict(extraction=[dict(question_id=q, **groups[q]) for q in ids[:125]],
        test=[dict(question_id=q, **groups[q]) for q in ids[125:]], seed=42, prompt=PROMPT,
        max_new_tokens=512, steps=128, block_length=32, decoder='fast_dllm', temperature=0.,
        audio_encoding=dict(mode='chunked-whisper-global-adapters-v1', chunk_seconds=30),
        layer=31, group='answer', strength=1,
        averaging='Mean angry-minus-neutral over matched forwards per question, then equal weight per question')


def compatible(row, allow_native=False):
    encoding = row.get('audio_encoding', {})
    modes = ('chunked-whisper-global-adapters-v1', 'native') if allow_native else ('chunked-whisper-global-adapters-v1',)
    return (row.get('seed') == 42 and row.get('prompt') == PROMPT
            and row.get('max_new_tokens') == 512 and row.get('decoder') == 'fast_dllm'
            and encoding.get('mode') in modes and encoding.get('chunk_seconds') == 30)


def reuse_answers(args, plan):
    path = args.benchmark/'run_config.json'
    if not path.exists():
        return
    config = json.loads(path.read_text())
    if not compatible(config):
        raise ValueError('Existing BBH benchmark uses a different prompt/seed/encoding recipe.')
    fingerprints = {(r['question_id'], r['voice'], r['style']): r['audio_sha256']
                    for r in config['recordings']}
    tasks = {(p['question_id'], p[s]['voice'], s): p[s] for p in plan['test'] for s in ('neutral', 'angry')}
    done = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(args.output/'results.jsonl')}
    with (args.output/'results.jsonl').open('a') as stream:
        for row in read_rows(args.benchmark/'results.jsonl'):
            key = row['question_id'], row['voice'], row['condition']
            if key in tasks and key not in done:
                if (not compatible(row, allow_native=True) or row['reference_answer'] != tasks[key]['answer']
                        or fingerprints.get(key) != tasks[key]['audio_sha256']):
                    raise ValueError('Existing BBH answer/audio differs from steering recipe.')
                stream.write(json.dumps(dict(row, reused_from=str(args.benchmark)))+'\n')
                done.add(key)
    cache = args.benchmark/'harness/upstream'
    if cache.exists() and not (args.output/'harness/upstream').exists():
        shutil.copytree(cache, args.output/'harness/upstream')


def summarize(args, destination=None):
    from score_bbh_harness import HarnessScorer
    plan = json.loads((args.output/'split.json').read_text())
    allowed = {p['question_id'] for p in plan['test']}
    bank = {q['question_id']: q for q in read_rows(args.source/'questions.jsonl')}
    rows = read_rows(args.output/'results.jsonl')
    if any(r['question_id'] not in allowed for r in rows):
        raise ValueError('Extraction question found in held-out results.')
    if 'harness_task' in plan:
        from rescore_audio_steering import HarnessTaskScorer
        scorer = HarnessTaskScorer(plan['harness_task'], args.output/'harness/upstream')
    else:
        scorer = HarnessScorer(args.output/'harness/upstream')
    values = scorer.score([r['response'] for r in rows],
        [dict(input=bank[r['question_id']]['original_input'], target=bank[r['question_id']]['original_target']) for r in rows])
    summary = dict(extraction_questions=125, test_questions=125, scorer=scorer.provenance, accuracy=[], paired=[])
    destination = Path(destination) if destination else args.output
    destination.mkdir(parents=True, exist_ok=True)
    rescored = [dict(row, harness={name: result[i] for name, result in values.items()})
                for i, row in enumerate(rows)]
    print('\nBBH: 125 extraction / 125 held-out test questions; L31 answer x1; seed42.')
    print('Same prompt and upstream harness strict/flexible scoring; completed pairs only. No classifier/outcome filter.')
    for metric, scores in values.items():
        groups = {c: {} for c in CONDITIONS}
        for row, result in zip(rows, scores):
            groups[row['condition']][row['question_id'], row['voice']] = result['correct']
        print(f'\n{metric}')
        for condition, answers in groups.items():
            n, correct = len(answers), sum(answers.values())
            summary['accuracy'].append(dict(metric=metric, condition=condition, n=n, correct=correct))
            print(f'  {condition}: {correct}/{n}' + (f' ({correct/n:.1%})' if n else ''))
        for condition in CONDITIONS[1:]:
            keys = sorted(groups['neutral'].keys() & groups[condition].keys())
            if not keys:
                continue
            result = paired([(groups['neutral'][k], groups[condition][k]) for k in keys])
            summary['paired'].append(dict(metric=metric, condition=condition, **result))
            lo, hi = result['ci95_gain_pp']
            print(f'  {condition} vs neutral: N={len(keys)}; gain={result["gain_pp"]:+.2f} pp; '
                  f'W/L={result["wins"]}/{result["losses"]}; p={result["mcnemar_exact_p"]:.4f}; CI=[{lo:+.1f}, {hi:+.1f}]')
    print('Exploratory split: aggregate delivery results on 200 of these questions were previously examined. P-values/intervals unadjusted.')
    write_json(destination/'summary.json', summary)
    (destination/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rescored))
    return summary


def run(args, plan):
    import numpy as np
    import soundfile as sf
    import torch
    from extract_vectors import gen_chunked, load_model

    pairs_dir = args.output/'pairs'; pairs_dir.mkdir(exist_ok=True)
    pending = [(i, p) for i, p in enumerate(plan['extraction']) if not (pairs_dir/f'{i:05d}.pt').exists()]
    complete = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(args.output/'results.jsonl', repair=True)}
    needed = [(p, c) for p in plan['test'] for c in CONDITIONS
              if (p['question_id'], p['neutral']['voice'], c) not in complete]
    print(f'{2*len(pending)} collection generations and {len(needed)} evaluation generations pending.', flush=True)
    if not pending and not needed:
        return
    cache = Path(os.environ.get('HF_HOME') or next((str(p) for p in
        (Path('/workspace/.hf_home'), Path('/workspace/hf-cache')) if (p/'hub').is_dir()), 'outputs/hf-diffa')).absolute()
    os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache/'hub'), HUGGINGFACE_HUB_CACHE=str(cache/'hub'))
    os.environ.pop('TRANSFORMERS_CACHE', None)
    model, tokenizer, features = load_model()
    layers = model.llm_model.model.transformer.blocks
    prompt = plan['prompt']

    def generate(task):
        path = Path(task['audio_file'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != task['audio_sha256']:
            raise ValueError('BBH audio changed.')
        audio, sr = sf.read(path, dtype='float32')
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        return gen_chunked(audio, sr, model, tokenizer, features, prompt, 512, 30)

    def capture(task):
        states = []
        def hook(module, inputs, result):
            states.append(result[0][0, -512:].float().mean(0).cpu())
        handle = layers[31].register_forward_hook(hook)
        try:
            response, _ = generate(task)
        finally:
            handle.remove()
        return response, torch.stack(states)

    for i, pair in pending:
        print(f'Collect [{i+1}/125] {pair["question_id"]} neutral + angry', flush=True)
        nr, neutral = capture(pair['neutral']); ar, angry = capture(pair['angry'])
        steps = min(len(neutral), len(angry))
        save_tensor(torch, dict(question_id=pair['question_id'], difference=(angry[:steps]-neutral[:steps]).mean(0),
            neutral_forwards=len(neutral), angry_forwards=len(angry), matched_forwards=steps,
            neutral_response=nr, angry_response=ar), pairs_dir/f'{i:05d}.pt')
    values = [torch.load(pairs_dir/f'{i:05d}.pt', map_location='cpu', weights_only=True) for i in range(125)]
    if any(v['question_id'] != p['question_id'] for v, p in zip(values, plan['extraction'])):
        raise ValueError('Pair cache differs from fixed split.')
    total = torch.stack([v['difference'] for v in values]).sum(0)
    sums = torch.zeros(len(layers), len(GROUPS), len(total)); sums[31, GROUPS.index('answer')] = total
    counts = torch.zeros(len(layers), dtype=torch.long); counts[31] = 125
    save_tensor(torch, dict(pair_mean_sums={'angry': sums}, pair_counts={'angry': counts}, groups=GROUPS,
        prompt=prompt, seed=42, max_new_tokens=512, method='mean', mode='fixed',
        collected_targets=[(31, 'answer')], extraction_question_ids=[p['question_id'] for p in plan['extraction']],
        averaging=plan['averaging'], audio_encoding=plan['audio_encoding']), args.output/'style-vectors.pt')
    directions = {(31, 'answer'): (total/125).to(device='cuda', dtype=torch.bfloat16)}
    bank = {q['question_id']: q for q in read_rows(args.source/'questions.jsonl')}
    for i, (pair, condition) in enumerate(needed, 1):
        task = pair['angry' if condition == 'angry' else 'neutral']
        print(f'Evaluate [{i}/{len(needed)}] {pair["question_id"]} {condition}', flush=True)
        interventions = [dict(layer=31, group='answer', strength=1)] if condition == CONDITION else []
        with install_interventions(layers, directions, interventions, {}, 512):
            response, encoding = generate(task)
        row = dict(question_id=pair['question_id'], voice=task['voice'], style=task['style'], condition=condition,
            audio_path=task['audio_path'], prompt=prompt, seed=42, max_new_tokens=512, decoder='fast_dllm',
            audio_encoding=encoding, response=response, reference_answer=bank[pair['question_id']]['answer'])
        with (args.output/'results.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(response, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['all', 'prepare', 'evaluate', 'summary'], default='all')
    p.add_argument('--existing-source', type=Path, default=Path('outputs/azure-bbh-logical200'))
    p.add_argument('--source', type=Path, default=Path('outputs/azure-bbh-logical250-steering'))
    p.add_argument('--benchmark', type=Path, default=Path('outputs/bbh-logical200-diffa2-harness-chunked'))
    p.add_argument('--output', type=Path, default=Path('outputs/bbh-logical250-angry-mean'))
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        summarize(args); return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode in ('all', 'prepare'):
            prepare(args)
        if args.mode == 'prepare':
            return
        plan = make_plan(args.source)
        path = args.output/'split.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('Frozen split/settings changed; use a new output folder.')
        write_json(path, plan)
        reuse_answers(args, plan)
        for kind in ('svamp', 'bbh'):
            subprocess.run([sys.executable, '-u', 'wait_for_experiments.py', '--kind', kind], check=True)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            run(args, plan)
        summarize(args)


if __name__ == '__main__':
    main()
