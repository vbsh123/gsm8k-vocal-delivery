"""Generate 500 SVAMP neutral/angry pairs and test delivery accuracy; steering is opt-in."""

import argparse
from decimal import Decimal
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

from compare_cloud_accuracy import PROMPT
from rescore_audio_steering import FLEXIBLE, harness_filter, numeric_equal, paired
from scoring import score_response
from steer import install_interventions, read_rows

SOURCE_REVISION = '7827fb803c3f6b017086f75024515b15bf31900f'
SOURCE_URL = f'https://raw.githubusercontent.com/arkilpatel/SVAMP/{SOURCE_REVISION}/SVAMP.json'
SOURCE_SHA256 = '5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e'
STEERED = 'neutral-GSM8K-L31-answer-a1'


def select_questions(data, count=500):
    rows = json.loads(data)
    random.Random(42).shuffle(rows)
    if not 1 <= count <= len(rows):
        raise ValueError('Question count outside the SVAMP dataset size.')
    return [dict(question_id='svamp_'+r['ID'], source_id=r['ID'],
        question=r['Body'].strip()+' '+r['Question'].strip(),
        answer=format(Decimal(str(r['Answer'])).normalize(), 'f'),
        original_answer=r['Answer'], equation=r['Equation'], problem_type=r['Type']) for r in rows[:count]]


def generate(args):
    args.source.mkdir(parents=True, exist_ok=True)
    source = args.source/'SVAMP-source.json'
    if not source.exists():
        with urlopen(SOURCE_URL, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
            raise ValueError('SVAMP source checksum changed.')
        source.write_bytes(data)
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('Cached SVAMP source checksum differs.')
    bank = select_questions(data, args.num_examples)
    text = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in bank)
    path = args.source/'questions.jsonl'
    if path.exists() and path.read_text() != text:
        raise ValueError('Selected questions changed; use a different source folder.')
    path.write_text(text)
    (args.source/'selection.json').write_text(json.dumps(dict(
        dataset='SVAMP official challenge set', source_url=SOURCE_URL, source_sha256=SOURCE_SHA256,
        seed=42, selected_ids=[r['source_id'] for r in bank],
        rule='Shuffle the full 1000-item source once; take first N, without model/classifier filtering'), indent=2)+'\n')
    print(f'{len(bank)} fixed SVAMP questions × neutral/angry = {2*len(bank)} Azure clips.', flush=True)
    command = [sys.executable, '-u', 'cloud_tts_smoke.py', '--provider', 'azure',
        '--azure-voice', 'en-US-JennyNeural', '--azure-intensity', '1.5',
        '--input-jsonl', str(path), '--num-examples', str(len(bank)), '--styles', 'neutral', 'angry',
        '--output', str(args.source), '--resume', '--retry-passes', '5']
    if args.mode == 'prepare':
        command.append('--prepare-only')
    subprocess.run(command, check=True)


def find_vectors(explicit):
    if explicit:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit.absolute()
    for config in [Path('outputs/steer-jenny-late-layers/original_mean/run_config.json'),
                   Path('outputs/steer-jenny-original-vector-L12-seed42/original_mean/run_config.json')]:
        if config.exists():
            values = json.loads(config.read_text())
            path = Path(values.get('settings', {}).get('vectors', 'MISSING'))
            if path.is_file():
                return path.absolute()
    for path in [Path('outputs/jenny-angry-vectors/style-vectors.pt'),
                 Path('outputs/steer-jenny-angry-L0-25-n50/jenny-angry-vectors/style-vectors.pt')]:
        if path.is_file():
            return path.absolute()
    raise FileNotFoundError('Original GSM8K vector not found. Pass --vectors /path/to/style-vectors.pt, '
                            'or --unsteered-only to evaluate only neutral/angry audio.')


def recordings(source, bank, steering):
    groups = {}; durations = []
    for row in read_rows(source/'plan.jsonl'):
        if row['provider'] != 'azure' or row['style'] not in ('neutral', 'angry'):
            continue
        question = bank[row['question_id']]
        if row['question'] != question['question'] or row['answer'] != question['answer']:
            raise ValueError('Audio plan differs from SVAMP source.')
        path = (source/row['audio_path']).absolute()
        with wave.open(str(path), 'rb') as audio:
            seconds = audio.getnframes()/audio.getframerate()
        key = row['question_id'], row['voice']
        group = groups.setdefault(key, {})
        if row['style'] in group:
            raise ValueError('Duplicate SVAMP delivery.')
        group[row['style']] = dict(row, audio_file=str(path),
            audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), seconds=seconds)
        durations.append(dict(question_id=row['question_id'], style=row['style'], seconds=seconds))
    tasks = []; excluded = []
    for (qid, _), group in groups.items():
        if set(group) != {'neutral', 'angry'}:
            raise ValueError(f'Missing neutral/angry pair: {qid}')
        if any(not 0 < row['seconds'] <= 30 for row in group.values()):
            excluded.append(qid)
            continue
        tasks.extend(dict(group[s], condition=s) for s in ('neutral', 'angry'))
        if steering:
            tasks.append(dict(group['neutral'], condition=STEERED))
    if not tasks:
        raise ValueError('No complete neutral/angry pairs fit 30 seconds.')
    return tasks, dict(rule='Both deliveries must fit 30 seconds; no cropping',
                      excluded_questions=excluded, durations=durations)


def score(response, gold, Filter):
    prediction = Filter(regex_pattern=FLEXIBLE, group_select=-1).apply([[response]], [{}])[0][0]
    normalize = lambda s: re.sub(r'\.$', '', s.replace(',', '').replace('$', '').strip()).lower()
    return dict(strict=score_response(response, gold)['correct'],
        harness_flexible=normalize(prediction) == normalize(gold),
        flexible_numeric=numeric_equal(prediction, gold), flexible_prediction=prediction)


def report(output, bank, Filter, include_steering=False):
    rows = [r for r in read_rows(output/'results.jsonl')
            if include_steering or r['condition'] in ('neutral', 'angry')]
    latest = {(r['question_id'], r['voice'], r['condition']): r for r in rows}
    groups = {}; scored = []
    for (qid, voice, condition), row in latest.items():
        if row['reference_answer'] != bank[qid]['answer']:
            raise ValueError('Saved reference differs from the official SVAMP source.')
        metrics = score(row['response'], bank[qid]['answer'], Filter)
        groups.setdefault(condition, {})[qid, voice] = metrics
        scored.append(dict(row, rescored=metrics))
    print('\nSVAMP: flexible_numeric is primary; GSM8K harness extraction + our numeric equivalence.')
    print('This is saved-answer scoring, not an official SVAMP harness inference task.')
    print(f'{"Condition":30} {"N":>5} {"Numeric accuracy":>17} {"Harness string":>16} {"Strict accuracy":>17}')
    totals = []
    for condition, values in groups.items():
        n = len(values)
        correct = {m: sum(v[m] for v in values.values()) for m in ('flexible_numeric', 'harness_flexible', 'strict')}
        print(f'{condition:30} {n:5} {correct["flexible_numeric"]/n:17.1%}'
              f' {correct["harness_flexible"]/n:16.1%} {correct["strict"]/n:17.1%}')
        totals.append(dict(condition=condition, n=n, correct=correct))
    comparisons = []
    base = groups.get('neutral', {})
    for condition, values in groups.items():
        if condition == 'neutral':
            continue
        keys = sorted(base.keys() & values.keys())
        if not keys:
            continue
        for metric in ('flexible_numeric', 'harness_flexible', 'strict'):
            result = paired([(base[k][metric], values[k][metric]) for k in keys])
            lo, hi = result['ci95_gain_pp']
            n = result['pairs']
            print(f'{condition} / {metric}: N={n}; neutral={result["baseline_correct"]/n:.1%}; '
                  f'test={result["steered_correct"]/n:.1%}; gain={result["gain_pp"]:+.2f} pp; '
                  f'W/L={result["wins"]}/{result["losses"]}; p={result["mcnemar_exact_p"]:.4f}; '
                  f'95% CI=[{lo:+.1f}, {hi:+.1f}] pp')
            comparisons.append(dict(condition=condition, metric=metric, **result))
    print('Matched questions and voice; no outcome/classifier filtering. P-values and intervals are unadjusted.')
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    (output/'summary.json').write_text(json.dumps(dict(accuracy=totals, paired=comparisons), indent=2)+'\n')


def evaluate(args, vectors, bank, Filter):
    tasks, eligibility = recordings(args.source, bank, vectors is not None)
    vector = dict(path=str(vectors), sha256=hashlib.sha256(vectors.read_bytes()).hexdigest()) if vectors else None
    config = dict(recordings=tasks, eligibility=eligibility, vector=vector,
        seed=42, prompt=PROMPT, max_new_tokens=512, decoder='fast_dllm', steps=128,
        block_length=32, temperature=0., primary_metric='flexible_numeric')
    path = args.output/'run_config.json'
    if path.exists():
        original = json.loads(path.read_text())
        if original != config:
            narrowed = dict(original, vector=None, recordings=[r for r in original['recordings']
                if r['condition'] in ('neutral', 'angry')])
            if vectors is not None or narrowed != config:
                raise ValueError('Settings or audio changed; use a separate result folder.')
            print('Resuming only neutral/angry; original three-condition config and all saved answers retained.', flush=True)
    else:
        path.write_text(json.dumps(config, indent=2)+'\n')
    done = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(args.output/'results.jsonl', repair=True)}
    pending = [r for r in tasks if (r['question_id'], r['voice'], r['condition']) not in done]
    print(f'{len(tasks)//(3 if vectors else 2)} matched questions; {len(pending)}/{len(tasks)} generations pending.', flush=True)
    print(f'Overlength/empty pairs excluded: {len(eligibility["excluded_questions"])}. Frozen vector: {vectors}', flush=True)
    if pending:
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True)
                time.sleep(30)
            run(config, pending, bank, args.output, Filter)
    report(args.output, bank, Filter, include_steering=vectors is not None)


def run(config, pending, bank, output, Filter):
    cache = Path(os.environ.get('HF_HOME') or next((str(p) for p in
        (Path('/workspace/.hf_home'), Path('/workspace/hf-cache')) if (p/'hub').is_dir()), 'outputs/hf-diffa')).absolute()
    os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache/'hub'), HUGGINGFACE_HUB_CACHE=str(cache/'hub'))
    os.environ.pop('TRANSFORMERS_CACHE', None)
    import numpy as np
    import soundfile as sf
    import torch
    from extract_vectors import gen, load_model
    directions = {}
    if config['vector']:
        state = torch.load(config['vector']['path'], map_location='cpu', weights_only=True)
        if state['prompt'] != PROMPT or state['max_new_tokens'] != 512 or state.get('mode', 'fixed') != 'fixed':
            raise ValueError('Expected the frozen original GSM8K mean vector and generation recipe.')
        if set(state.get('extraction_question_ids', [])) & set(bank):
            raise ValueError('Evaluation/extraction overlap.')
        count = state['pair_counts']['angry'][31]
        if count <= 0:
            raise ValueError('No L31 direction in the selected vector.')
        direction = state['pair_mean_sums']['angry'][31, state['groups'].index('answer')]/count
        directions[31, 'answer'] = direction.to(device='cuda', dtype=torch.bfloat16)
        del state
    model, tokenizer, features = load_model()
    layers = model.llm_model.model.transformer.blocks
    for index, task in enumerate(pending, 1):
        path = Path(task['audio_file'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != task['audio_sha256']:
            raise ValueError('SVAMP audio changed.')
        audio, sr = sf.read(path, dtype='float32')
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        intervention = [dict(layer=31, group='answer', strength=1)] if task['condition'] == STEERED else []
        print(f'[{index}/{len(pending)}] {task["question_id"]} {task["condition"]}', flush=True)
        started = time.monotonic()
        with install_interventions(layers, directions, intervention, {}, 512):
            response = gen(audio, sr, model, tokenizer, features, PROMPT, 512)
        metrics = score(response, bank[task['question_id']]['answer'], Filter)
        row = dict(question_id=task['question_id'], voice=task['voice'], style=task['style'],
            audio_path=task['audio_path'], condition=task['condition'], response=response,
            reference_answer=bank[task['question_id']]['answer'], seed=42, prompt=PROMPT,
            max_new_tokens=512, decoder='fast_dllm', generation_seconds=time.monotonic()-started, **metrics)
        with (output/'results.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(f'  numeric_correct={metrics["flexible_numeric"]}; strict_correct={metrics["strict"]}\n{response}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['all', 'prepare', 'generate', 'evaluate', 'summary'], default='all')
    p.add_argument('--source', type=Path, default=Path('outputs/azure-svamp500'))
    p.add_argument('--output', type=Path, default=Path('outputs/svamp500-diffa2'))
    p.add_argument('--num-examples', type=int, default=500)
    p.add_argument('--vectors', type=Path, help='Frozen original GSM8K mean; otherwise find the existing winner checkpoint')
    p.add_argument('--steer', action='store_true', help='Explicitly add frozen GSM8K L31 steering (default: only neutral/angry)')
    p.add_argument('--unsteered-only', action='store_true', help='Only neutral/angry delivery accuracy, no steering')
    args = p.parse_args()
    if args.steer and args.unsteered_only:
        p.error('Choose either --steer or --unsteered-only.')
    if args.vectors and not args.steer:
        p.error('--vectors requires explicit --steer.')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        Filter = harness_filter(args.output/'upstream')
        report(args.output, {r['question_id']: r for r in read_rows(args.source/'questions.jsonl')}, Filter,
               include_steering=args.steer)
        return
    with (args.output/'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        vectors = find_vectors(args.vectors) if args.mode in ('all', 'evaluate') and args.steer else None
        if args.mode in ('all', 'prepare', 'generate'):
            generate(args)
        if args.mode in ('all', 'evaluate'):
            Filter = harness_filter(args.output/'upstream')
            bank = {r['question_id']: r for r in read_rows(args.source/'questions.jsonl')}
            evaluate(args, vectors, bank, Filter)


if __name__ == '__main__':
    main()
