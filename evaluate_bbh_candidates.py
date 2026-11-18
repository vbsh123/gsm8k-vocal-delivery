"""BBH-three neutral/angry evaluation for Phi-4, MiniCPM-o 4.5 or Audio Flamingo 3.

Separate original/explicit prompt results. No collection, steering or layer search.
"""
import argparse
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from bbh_candidate_adapters import MODELS, make_generator
import evaluate_bbh_audio_models as existing
from kimi_bbh_layer_diagnostic import digest, freeze
from steer import read_rows

SOURCE_CANDIDATES = [Path('outputs')/s for s in ('azure-bbh-three-models-regenerated',
    'azure-bbh-logical250-steering', 'azure-bbh-logical200')]


def source_folder(explicit=None):
    if explicit is not None:
        return explicit
    for path in SOURCE_CANDIDATES:
        if len(read_rows(path/'questions.jsonl')) == 250:
            return path
    return SOURCE_CANDIDATES[0]


def audio(source):
    questions = read_rows(source/'questions.jsonl')
    if questions and len(questions) != 250:
        raise ValueError('Need all 250 BBH-three questions. Use a separate source folder for regenerated full audio.')
    if questions and (source/'plan.jsonl').exists():
        try:
            existing.make_plan(source, 'moss')
        except FileNotFoundError:
            pass  # A valid existing request plan may still have missing WAVs.
        else:
            print('All 500 neutral/angry clips verified; reusing source without rewriting its request plan.', flush=True)
            return
    # Existing helper is CPU-only, verifies upstream source and resumes 500 Azure clips.
    existing.generate_audio(source)


def make_plan(source, model, variant):
    if model not in MODELS or variant not in ('original', 'explicit'):
        raise ValueError('Unknown model or prompt variant.')
    plan = existing.make_plan(source, 'moss')  # Reuse ONLY source/audio validation; no MOSS imports or loading.
    if len(plan['bank']) != 250:
        raise ValueError('Expected all 250 original BBH-three questions.')
    spec = MODELS[model]
    if model == 'flamingo' and any(r['seconds'] > 600 for r in plan['recordings']):
        raise ValueError('Audio Flamingo has a native 600-second cap; refusing to crop input.')
    plan.update(prompt=existing.PROMPT if variant == 'original' else existing.EXPLICIT_REASONING_PROMPT,
        prompt_variant=variant, model_key=model, protocol='bbh-three-candidate-audio-v1',
        recipe=dict(spec, dtype='bfloat16', do_sample=False, num_beams=1, use_cache=True,
                    repetition_penalty=1., text_only_output=True),
        audio_encoding='Full mono waveform resampled to 16kHz, then native model processor/encoder; no custom feature joining.',
        caveat='Audio-adapted inference; upstream BBH task filters and metric on verbatim answers. '
               'Original and explicit prompts are separate experiments, not pooled evidence.',
        duration_audit=dict(max_seconds=max(r['seconds'] for r in plan['recordings']),
            over_40_seconds=sum(r['seconds'] > 40 for r in plan['recordings'])))
    return plan


def prepare(args):
    plans = []
    for variant in ('original', 'explicit') if args.prompt == 'both' else (args.prompt,):
        output = args.output/args.model/variant; output.mkdir(parents=True, exist_ok=True)
        plan = make_plan(source_folder(args.source), args.model, variant)
        if getattr(args, 'concurrent', False):
            plan['resource_policy'] = dict(mode='concurrent-opt-in', gpu_budget_gib=args.gpu_budget_gib,
                free_headroom_gib=2, note='Shares GPU with an existing job; runtimes are not comparable to isolated runs. '
                'PyTorch allocator cap excludes external CUDA allocations.')
        freeze(output/'run_config.json', plan)
        plans.append((output, plan))
        print(f'{args.model}/{variant}: 250 questions, 500 neutral/angry answers, no steering. '
              f'Max audio={plan["duration_audit"]["max_seconds"]:.2f}s.', flush=True)
    return plans


def pending(plans):
    jobs = []
    for output, plan in plans:
        done = {}
        for row in read_rows(output/'results.jsonl', repair=True):
            if (row['prompt'] != plan['prompt'] or row['model'] != plan['recipe']['model']
                    or row['revision'] != plan['recipe']['revision'] or row.get('seed') != 42
                    or row.get('max_new_tokens') != 512 or row.get('interventions') != []):
                raise ValueError('Saved answer settings differ from frozen candidate experiment.')
            key = row['question_id'], row['voice'], row['style']
            if key in done and done[key]['response'] != row['response']:
                raise ValueError('Conflicting completed responses.')
            done[key] = row
        for task in plan['recordings']:
            key = task['question_id'], task['voice'], task['style']
            if key in done:
                if done[key]['audio_sha256'] != task['audio_sha256']:
                    raise ValueError('Saved answer refers to different audio.')
            else:
                jobs.append((output, plan, task))
    return jobs


def concurrent_budget(budget_gib):
    """Reserve headroom before loading MiniCPM; cap this process's Torch allocator."""
    import torch
    required = (budget_gib+2)*2**30
    while True:
        free, total = torch.cuda.mem_get_info(0)
        if budget_gib*2**30 >= total:
            raise ValueError('GPU budget must be smaller than total GPU memory.')
        if free >= required:
            break
        print(f'Waiting for concurrent GPU room: {free/2**30:.1f} GiB free; '
              f'{budget_gib+2:.1f} GiB required including headroom.', flush=True)
        time.sleep(30)
    torch.cuda.set_per_process_memory_fraction(budget_gib*2**30/total, device=0)
    print(f'Concurrent mode: PyTorch allocator capped at {budget_gib:.1f} GiB; '
          f'{free/2**30:.1f} GiB free before loading. Existing jobs left running.', flush=True)


def concurrent_budget_now(budget_gib):
    """Explicit concurrent launch: cap memory and fail rather than wait for GPU."""
    import torch
    free, total = torch.cuda.mem_get_info(0)
    if budget_gib <= 0 or budget_gib*2**30 >= total: raise ValueError('Invalid concurrent GPU budget.')
    if free < (budget_gib+2)*2**30:
        raise RuntimeError(f'Need{budget_gib+2:.1f} GiB free for concurrent startup; have{free/2**30:.1f}. No GPU queue wait or jobs stopped.')
    torch.cuda.set_per_process_memory_fraction(budget_gib*2**30/total, device=0)
    print(f'Concurrent startup: {free/2**30:.1f} GiB free; Torch cap{budget_gib:.1f} GiB; no GPU queue.', flush=True)


def run(plans, model_key, limit=0):
    jobs = pending(plans)
    if limit:
        jobs = jobs[:limit]
    print(f'{len(jobs)} generations pending in this invocation.', flush=True)
    if not jobs:
        return
    import numpy as np
    import torch
    actual = version('transformers')
    if actual != MODELS[model_key]['transformers']:
        raise RuntimeError('Use the separate pinned environment: bash setup_bbh_candidate.sh '+model_key)
    generate = make_generator(model_key)
    for output, plan in plans:
        (output/'runtime.json').write_text(json.dumps(dict(python=sys.version, torch=torch.__version__,
            transformers=actual, numpy=np.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
            token_count='retokenized text' if model_key == 'minicpm' else 'generated suffix token IDs',
            resource_policy=plan.get('resource_policy', {'mode':'exclusive'}),
            generation='native greedy text-only, no interventions'), indent=2)+'\n')
    for i, (output, plan, task) in enumerate(jobs, 1):
        if digest(task['audio_file']) != task['audio_sha256']:
            raise ValueError('Recording changed after preparation.')
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        print(f'[{i}/{len(jobs)}] {model_key}/{plan["prompt_variant"]} {task["question_id"]} {task["style"]}', flush=True)
        started = time.monotonic()
        response, tokens = generate(task, plan['prompt'])
        if not isinstance(response, str):
            raise TypeError('Expected a raw text response.')
        row = dict(task, response=response, response_tokens=tokens, prompt=plan['prompt'],
            seed=42, model=plan['recipe']['model'], revision=plan['recipe']['revision'],
            max_new_tokens=512, generation_seconds=time.monotonic()-started, interventions=[])
        with (output/'results.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n'); stream.flush()
        print(response, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=tuple(MODELS), required=True)
    p.add_argument('--mode', choices=('audio', 'prepare', 'evaluate', 'summary'), default='evaluate')
    p.add_argument('--prompt', choices=('original', 'explicit', 'both'), default='both')
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/bbh-three-candidates'))
    p.add_argument('--max-generations', type=int, default=0)
    p.add_argument('--concurrent', action='store_true', help='Opt in to shared-GPU MiniCPM; own queue lock remains active.')
    p.add_argument('--gpu-budget-gib', type=float, default=22., help='Concurrent-mode PyTorch allocator cap; requires another 2 GiB free.')
    args = p.parse_args()
    if args.max_generations < 0:
        p.error('--max-generations must be nonnegative')
    if args.concurrent and args.model != 'minicpm':
        p.error('Concurrent mode is currently prepared only for MiniCPM.')
    if args.gpu_budget_gib <= 0:
        p.error('GPU budget must be positive.')
    if args.mode == 'audio':
        audio(source_folder(args.source)); return
    if args.mode == 'summary':
        for variant in ('original', 'explicit') if args.prompt == 'both' else (args.prompt,):
            existing.report(args.output/args.model/variant)
        return
    plans = prepare(args)
    if args.mode == 'prepare':
        return
    root = args.output/args.model
    with (root/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.concurrent:
            concurrent_budget(args.gpu_budget_gib)
            run(plans, args.model, args.max_generations)
            return
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            run(plans, args.model, args.max_generations)


if __name__ == '__main__':
    main()
