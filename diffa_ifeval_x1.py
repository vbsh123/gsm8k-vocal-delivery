"""Complete the original DIFFA spoken IFEval L31 answer-position experiment."""
import argparse
import json
import os
from pathlib import Path

import run_spoken_ifeval as official
import winning_diffa_sweeps as winners
from extract_audio_vectors import GROUPS, save_tensor
from kimi_spoken_ifeval import SPLIT, audio_folder
from phi4_queued_workers import gpu_queue
from phi4_sweep import job_lock
from rescore_audio_steering import paired
from steer import read_rows

DATA = Path('data/ifeval_spoken_steering')
ROOT = Path('outputs/diffa-ifeval-spoken-x1')
STYLES = ('sad', 'happy', 'terrified')
BUDGET = 1024
VOICE = 'en-US-JennyNeural'


def inputs():
    split = json.loads(SPLIT.read_text())
    ref = json.loads((DATA/'reference.json').read_text())
    fit, test = split['collection_ids'], split['test_ids']
    if (len(fit) != 254 or len(test) != 254 or len(set(fit + test)) != 508
            or set(ref['scores']) != set(test) or set(ref['audio']) != set(fit + test)
            or ref['prompt'] != official.PROMPT or ref['revision'] != official.REVISION):
        raise ValueError('Original spoken IFEval split or reference mismatch.')
    return fit, test, ref


def recordings(style, ids, ref):
    terrified = style == 'terrified'
    env = 'IFEVAL_TERRIFIED_AUDIO' if terrified else 'IFEVAL_AUDIO_DIR'
    subtree = 'jenny-terrified-ifeval/spoken' if terrified else 'azure-spoken-ifeval'
    prefix = 'jenny-terrified-math-ifeval' if terrified else 'jenny'
    names = [ref['audio'][q]['file'].replace('-neutral.wav', f'-{style}.wav') for q in ids]
    cache = ROOT/'audio'/style
    candidates = [Path(os.environ[env])] if os.environ.get(env) else [
        cache, Path('outputs')/subtree,
        *(parent/name for parent in (Path('/workspace'), Path.cwd(), Path.home(), Path('outputs'))
          for name in (prefix, prefix+'.zip'))]
    if not terrified and not os.environ.get(env):
        for parent in (Path('/workspace'), Path.cwd(), Path.home(), Path('outputs')):
            candidates.extend(sorted(parent.glob('jenny-six-datasets-audio-*.zip')))
    source = next((p for p in candidates if p.exists() and
                   (p != cache or all((p/n).is_file() for n in names))), None)
    if source is None:
        raise FileNotFoundError(f'Set {env} to the full spoken IFEval WAV directory or Jenny archive ({prefix}).')
    folder = audio_folder(source, names, cache, subtree)
    result = {}
    for q, name in zip(ids, names):
        path = folder/name
        sha = ref['audio'][q][style]
        if winners.digest(path) != sha:
            raise ValueError(f'Archived audio hash differs for {q}/{style} at {path}')
        result[q] = dict(question_id=q, voice=VOICE, audio_file=str(path.resolve()), audio_sha256=sha)
    return result


def fit_vector(style, folder, ids, ref, neutral, emotion, model, tokenizer, features):
    import torch
    from diffa_happy_nightly import generate
    plan = dict(max_new_tokens=BUDGET)
    path = folder/'style-vectors.pt'
    if path.exists():
        return path
    previous = Path(f'outputs/diffa-{style}-winners/ifeval')
    prev_vector = previous/'style-vectors.pt'
    state_path = folder/'fit.pt'
    completed = []
    total = torch.zeros(model.llm_model.get_input_embeddings().weight.shape[1], dtype=torch.float64)
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        completed, total = state['ids'], state['sum']
    elif (previous/'collection-stream.pt').exists() or prev_vector.exists():
        cfg = json.loads((previous/'run_config.json').read_text())
        if (cfg['emotion'] != style or cfg['collection_prompt'] != official.PROMPT
                or cfg['max_new_tokens'] != BUDGET or cfg['seed'] != 42
                or set(cfg['collection_ids']) != set(ids)):
            raise ValueError('Historical collection has different settings.')
        for pair in cfg['pairs']:
            q = pair['question_id']
            if any(pair[key]['audio_sha256'] != ref['audio'][q][label]
                   for key, label in [('neutral','neutral'),('emotion',style)]):
                raise ValueError(f'Historical collection WAV mismatch at {q}.')
        if prev_vector.exists():
            return prev_vector
        from diffa_stream_collection import load
        state = load(cfg, previous, winners.digest(previous/'run_config.json'))
        completed = state['completed_ids']
        if completed:
            total = state['difference_sum'][31, GROUPS.index('answer')].double().clone()
    if len(set(completed)) != len(completed) or not set(completed) <= set(ids):
        raise ValueError('Invalid fitting checkpoint.')
    print(f'{style} resuming with {len(completed)}/{len(ids)} pairs.', flush=True)

    def capture(record):
        states = []
        def observe(module, args, result):
            states.append(result[0][0, -BUDGET:].detach().float().mean(0).cpu())
        hook = model.llm_model.model.transformer.blocks[31].register_forward_hook(observe)
        try:
            generate(record, plan, official.PROMPT, model, tokenizer, features)
        finally:
            hook.remove()
        return torch.stack(states)

    for q in ids:
        if q in completed:
            continue
        n, e = capture(neutral[q]), capture(emotion[q])
        steps = min(len(n), len(e))
        total += (e[:steps] - n[:steps]).mean(0).double()
        completed.append(q)
        save_tensor(torch, dict(ids=completed, sum=total), state_path)
        print(f'{style} fit {len(completed)}/{len(ids)} {q}', flush=True)
    sums = torch.zeros(32, 1, total.numel())
    sums[31, 0] = total.float()
    counts = torch.zeros(32, dtype=torch.long)
    counts[31] = len(ids)
    save_tensor(torch, dict(pair_mean_sums={style:sums}, pair_counts={style:counts}, groups=['answer'],
                            method='mean', mode='fixed', prompt=official.PROMPT, max_new_tokens=BUDGET,
                            seed=42, extraction_question_ids=ids), path)
    return path


def scored_results(folder, ids, ref, bank, evaluator):
    rows = read_rows(folder/'results.jsonl')
    if not rows:
        return {}
    job = json.loads((folder/'job.json').read_text())
    by_id = {}
    for row in rows:
        q = row['question_id']
        if (q not in ids or q in by_id or row['condition'] != 'L31-answer-a1'
                or row['audio_sha256'] != ref['audio'][q]['neutral']
                or row['vector_sha256'] != job['source']['vector_sha256']
                or row['prompt'] != official.PROMPT or row['steered_forwards'] <= 0):
            raise ValueError(f'Invalid held-out steering row in {folder}: {q}')
        score = official.score(bank[q], row['response'], evaluator)
        by_id[q] = {mode: score[mode]['prompt'] for mode in ('strict','loose')}
    return by_id


def summary(fit, test, ref, bank, evaluator):
    scores = {'angry': {q:ref['scores'][q]['angry'] for q in test}}
    for style in STYLES:
        folder = ROOT/style
        scores[style] = scored_results(folder, set(test), ref, bank, evaluator)
        if style == 'sad' or (folder/'job.json').exists() or (folder/'style-vectors.pt').exists():
            fitted = len(fit)
        elif (folder/'fit.pt').exists():
            import torch
            fitted = len(torch.load(folder/'fit.pt', map_location='cpu', weights_only=True)['ids'])
        else:
            fitted = 0
        print(f'{style}: {fitted}/{len(fit)} fitting pairs, {len(scores[style])}/{len(test)} held-out responses.')
    comparisons = []
    for style, values in scores.items():
        if not values:
            continue
        for mode in ('strict','loose'):
            stats = paired([(ref['scores'][q]['neutral'][mode], result[mode]) for q,result in values.items()])
            print(f'{style} x1 {mode}: {stats["baseline_correct"]}/{stats["pairs"]} -> '
                  f'{stats["steered_correct"]}/{stats["pairs"]}, {stats["gain_pp"]:+.2f} pp, '
                  f'95% CI {stats["ci95_gain_pp"]}, exact McNemar p={stats["mcnemar_exact_p"]:.4g}')
            comparisons.append(dict(style=style, mode=mode, **stats))
    winners.atomic_text(ROOT/'summary.json', json.dumps(comparisons, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['run','summary','prepare'])
    args = p.parse_args()
    fit, test, ref = inputs()
    ROOT.mkdir(parents=True, exist_ok=True)
    package = official.setup_official()
    bank = {f"ifeval_{r['key']}":r for r in read_rows(package/'data/input_data.jsonl')}
    evaluator = official.official_scorer()
    if args.action == 'summary':
        summary(fit, test, ref, bank, evaluator)
        return
    with job_lock(ROOT):
        neutral = recordings('neutral', fit + test, ref)
        clips = {s:recordings(s, fit, ref) for s in ('happy','terrified')}
        sad = DATA/'sad.pt'
        if winners.digest(sad) != ref['sad_sha256']:
            raise ValueError('Archived sad checkpoint hash mismatch.')
        winners.freeze(ROOT/'protocol.json', dict(reference=winners.digest(DATA/'reference.json'),
                       split=winners.digest(SPLIT), layer=31, group='answer', strength=1,
                       seed=42, budget=BUDGET, prompt=official.PROMPT, model='DIFFA-2'))
        print('Verified spoken IFEval WAVs, original neutral scores and sad checkpoint.', flush=True)
        if args.action == 'prepare':
            return
        for q in test:
            official.score(bank[q], 'Testing the scorer.', evaluator)
        from extract_vectors import load_model
        with gpu_queue():
            model, tokenizer, features = load_model()
            for style in STYLES:
                folder = ROOT/style
                folder.mkdir(exist_ok=True)
                vector = sad if style == 'sad' else fit_vector(
                    style, folder, fit, ref, neutral, clips[style], model, tokenizer, features)
                _, info = winners.read_vector(vector, style)
                if (set(info['extraction_ids']) != set(fit) or len(info['extraction_ids']) != len(fit)
                        or info['pairs'] != len(fit) or info['collection_prompt'] != official.PROMPT
                        or info['collection_tokens'] != BUDGET):
                    raise ValueError(f'{style} vector differs from the original IFEval fitting recipe.')
                job = dict(source=dict(name='ifeval',emotion=style,vector=str(vector.resolve()),
                                       vector_sha256=winners.digest(vector)),
                           target=dict(name='ifeval',records=[neutral[q] for q in test],
                                       prompt=official.PROMPT,max_new_tokens=BUDGET),
                           layer=31,group='answer',strengths=[1.])
                winners.freeze(folder/'job.json', job)
                scored_results(folder, set(test), ref, bank, evaluator)
                print(f'{style}: steering held-out neutral audio.', flush=True)
                winners.generate_job(job, folder, model, tokenizer, features)
        summary(fit, test, ref, bank, evaluator)


if __name__ == '__main__':
    main()
