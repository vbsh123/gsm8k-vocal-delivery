"""Collect matched angry-minus-neutral directions by decoding phase and 20-forward window."""

import argparse
import json
from pathlib import Path
import random

from extract_audio_vectors import GROUPS, save_tensor
from steer import diffusion_progress

TARGETS = [(11, 'answer'), (12, 'acoustic'), (16, 'answer')]


def paired_bins(neutral, angry, neutral_phases, angry_phases, torch, window_size=20):
    steps = min(len(neutral), len(angry))
    neutral, angry = neutral[:steps], angry[:steps]
    difference = angry - neutral
    windows = torch.stack([difference[i:i+window_size].mean(0) for i in range(0, steps, window_size)])
    phases, valid = [], []
    for phase in range(3):
        n = neutral[torch.tensor(neutral_phases[:steps]) == phase]
        a = angry[torch.tensor(angry_phases[:steps]) == phase]
        count = min(len(n), len(a))
        phases.append((a[:count] - n[:count]).mean(0) if count else torch.zeros_like(difference[0]))
        valid.append(bool(count))
    return dict(mean=difference.mean(0), phase=torch.stack(phases), phase_valid=valid,
                forward_windows=windows, matched_forwards=steps)


def collect(plan, output, torch):
    import numpy as np
    import soundfile as sf
    import src.fast_dllm.diffa_generate as decoder
    from extract_vectors import gen, load_model

    pending = [(i, pair) for i, pair in enumerate(plan['pairs']) if not (output/'pairs'/f'{i:05d}.pt').exists()]
    print(f'{len(plan["pairs"])} pairs; {2*len(pending)} generations pending; matched forwards; seed42.', flush=True)
    if not pending:
        return
    model, tokenizer, features = load_model()
    targets = plan.get('targets', TARGETS)
    bounds, captured, phases = {}, [[] for _ in targets], []
    original = decoder.insert_audio_embedding

    def insert(text, speech, lengths):
        start = int(lengths[0].item())
        bounds.update(start=start, acoustic_end=start+model.config.prompt_size, end=start+speech.shape[1])
        return original(text, speech, lengths)

    def hook_for(index, group):
        def hook(module, inputs, result):
            positions = slice(-512, None) if group == 'answer' else slice(bounds['start'], bounds['acoustic_end'])
            captured[index].append(result[0][0, positions].float().mean(0).cpu())
            if index == 0:
                phases.append(progress['phase'])
        return hook

    handles = [model.llm_model.model.transformer.blocks[layer].register_forward_hook(hook_for(i, group))
               for i, (layer, group) in enumerate(targets)]
    decoder.insert_audio_embedding = insert

    def capture(cid):
        nonlocal captured, phases, progress
        captured, phases = [[] for _ in targets], []
        audio, rate = sf.read(plan['manifest'][cid]['audio_path'], dtype='float32')
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        with diffusion_progress(model) as progress:
            response = gen(audio, rate, model, tokenizer, features, plan['prompt'], 512)
        return response, torch.stack([torch.stack(x) for x in captured], dim=1), phases

    progress = {}
    try:
        for i, pair in pending:
            print(f'[{i+1}/{len(plan["pairs"])}] {pair["question_id"]} neutral + angry', flush=True)
            nr, n, neutral_phases = capture(pair['neutral_id'])
            ar, a, ap = capture(pair['angry_id'])
            row = paired_bins(n, a, neutral_phases, ap, torch)
            row.update(pair=pair, neutral_response=nr, angry_response=ar,
                       neutral_forwards=len(n), angry_forwards=len(a))
            save_tensor(torch, row, output/'pairs'/f'{i:05d}.pt')
            print(f'  neutral={len(n)} angry={len(a)} matched={row["matched_forwards"]}; saved.', flush=True)
    finally:
        decoder.insert_audio_embedding = original
        for handle in handles:
            handle.remove()


def fit(plan, output, torch, modes=('phase', 'forward_windows')):
    targets = plan.get('targets', TARGETS)
    rows = [torch.load(output/'pairs'/f'{i:05d}.pt', map_location='cpu', weights_only=True)
            for i in range(len(plan['pairs']))]
    width = rows[0]['mean'].shape[-1]
    excluded = sorted({r['question_id'] for r in plan['manifest'].values()})
    for mode in modes:
        bins = 3 if mode == 'phase' else max(len(r[mode]) for r in rows)
        total = torch.zeros(32, len(GROUPS), width)
        sums = torch.zeros(32, bins, len(GROUPS), width)
        counts = torch.zeros(32, bins, dtype=torch.long)
        pair_counts = torch.zeros(32, dtype=torch.long)
        for row, pair in zip(rows, plan['pairs']):
            if row['pair'] != pair:
                raise ValueError('Saved temporal pair differs from collection plan')
            for index, (layer, group) in enumerate(targets):
                g = GROUPS.index(group)
                total[layer, g] += row['mean'][index]
                pair_counts[layer] += 1
                for b, value in enumerate(row[mode]):
                    if mode == 'phase' and not row['phase_valid'][b]:
                        continue
                    sums[layer, b, g] += value[index]
                    counts[layer, b] += 1
        folder = output/mode
        folder.mkdir(exist_ok=True)
        metadata = dict(mode=mode, window_size=20)
        state = dict(pair_mean_sums={'angry': total}, pair_counts={'angry': pair_counts},
                     temporal_sums={'angry': sums}, temporal_counts={'angry': counts},
                     groups=GROUPS, styles=['angry'], prompt=plan['prompt'], max_new_tokens=512,
                     seed=42, seed_policy='constant_before_every_generation', pairs=plan['pairs'],
                     extraction_question_ids=excluded, collected_targets=targets,
                     averaging='matched; equal weight per contributing question', **metadata)
        checkpoint = folder/'style-vectors.pt'
        if not checkpoint.exists():
            save_tensor(torch, state, checkpoint)
        (folder/'direction-settings.json').write_text(json.dumps(metadata, indent=2)+'\n')
        (folder/'extraction-questions.jsonl').write_text(''.join(json.dumps(dict(question_id=q))+'\n' for q in excluded))
        contributors = counts[targets[0][0]].tolist()
        (folder/'contributors.json').write_text(json.dumps(contributors)+'\n')
        print(f'{mode}: contributing pairs per bin: {contributors}. Missing/beyond-range bins use the overall matched mean.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', type=Path, default=Path('outputs/jenny-svd-probe'))
    p.add_argument('--original-vector', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('outputs/jenny-large/temporal-vectors'))
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--targets', nargs='+', default=['11:answer','12:acoustic','16:answer'],
                   help='Layer:group targets; one group per layer, acoustic or answer.')
    p.add_argument('--modes', nargs='+', choices=['phase','forward_windows'],default=['phase','forward_windows'])
    args = p.parse_args()
    try:
        targets = [(int(t.split(':')[0]), t.split(':')[1]) for t in args.targets]
        if any(not 0<=layer<32 or group not in ('answer','acoustic') for layer,group in targets) or len({l for l,g in targets})!=len(targets):
            raise ValueError()
    except (ValueError,IndexError):
        p.error('Use targets such as 31:answer; layers 0–31, acoustic/answer, one group per layer.')
    import torch
    source = json.loads((args.collection/'collection-config.json').read_text())
    if source.get('averaging', 'matched') != 'matched':
        p.error('Use the matched-forward collection')
    original = torch.load(args.original_vector, map_location='cpu', weights_only=True, mmap=True)
    keys = ['question_id', 'speaker', 'neutral_id', 'angry_id']
    if ([{k: r[k] for k in keys} for r in original['pairs']] !=
            [{k: r[k] for k in keys} for r in source['pairs']]):
        p.error('Temporal extraction must use the same pairs as the original vector')
    plan = dict(pairs=source['pairs'], manifest=source['manifest'], prompt=source['prompt'],
                targets=targets, seed=42, window_size=20, averaging='matched')
    plan = json.loads(json.dumps(plan))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'collection-config.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        p.error('Temporal collection changed; use another output folder')
    path.write_text(json.dumps(plan, indent=2)+'\n')
    (args.output/'pairs').mkdir(exist_ok=True)
    if not args.prepare_only:
        collect(plan, args.output, torch)
        fit(plan, args.output, torch, args.modes)


if __name__ == '__main__':
    main()
