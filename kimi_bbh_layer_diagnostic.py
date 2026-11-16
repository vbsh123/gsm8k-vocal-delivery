"""Recovery-only Kimi BBH-3 layer diagnostic, NOT a held-out accuracy experiment.

Collect AUDIO and ANSWER on 125 pairs; steer each on 30 disjoint recoveries.
Compare with Instruct-minus-base descriptively, on identical encoded inputs and
teacher-forced answer histories. Never steer with the checkpoint difference.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from unittest.mock import patch

from steer import read_rows

BASE_MODEL = 'moonshotai/Kimi-Audio-7B'
BASE_REVISION = 'd90be7113cd5be6bcf54fae2aabcad060a23f6cf'
LAYERS = tuple(range(28))
PROTOCOL = 'kimi-bbh3-collect125-audio-answer-test30-v3'
GROUPS = ('audio', 'answer')
PARAMS = dict(output_type='text', max_new_tokens=512, text_temperature=0., audio_temperature=0.,
              text_top_k=5, audio_top_k=5, text_repetition_penalty=1., audio_repetition_penalty=1.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Settings changed: {path}. Use a different output directory.')
    path.write_text(json.dumps(value, indent=2)+'\n')


def matched_pairs(rows):
    groups = {}
    for row in rows:
        if row['style'] not in ('neutral', 'angry'):
            continue
        key = row['question_id'], row['voice']
        group = groups.setdefault(key, {})
        if row['style'] in group:
            raise ValueError(f'Duplicate scored recording: {key}, {row["style"]}')
        group[row['style']] = row
    pairs = []
    for key, group in sorted(groups.items()):
        if set(group) == {'neutral', 'angry'}:
            pairs.append(dict(question_id=key[0], voice=key[1], **group))
    return pairs


def recovery_pairs(rows):
    """All flexible-score recoveries, not net wins and not strict-format wins."""
    return [p for p in matched_pairs(rows)
            if p['angry']['harness']['flexible-extract']['correct']
            and not p['neutral']['harness']['flexible-extract']['correct']]


def collection_pairs(rows, test, count=125):
    excluded = {p['question_id'] for p in test}
    candidates = [p for p in matched_pairs(rows) if p['question_id'] not in excluded]
    random.Random(42).shuffle(candidates)
    if len(candidates) < count:
        raise ValueError(f'Need {count} collection pairs excluding recovery test questions; have {len(candidates)}.')
    return candidates[:count]


def prepare(benchmark, output):
    from evaluate_kimi_audio import MODEL, REVISION, SOURCE_REVISION, TOKENIZER_MODEL, TOKENIZER_REVISION
    previous = json.loads((benchmark/'run_config.json').read_text())
    if previous['recipe']['model'] != MODEL or previous['recipe']['revision'] != REVISION:
        raise ValueError('Expected the pinned Kimi-Instruct benchmark.')
    if previous['max_new_tokens'] != 512 or previous['seed'] != 42:
        raise ValueError('Original generation settings differ from this diagnostic.')
    rows = read_rows(benchmark/'rescored.jsonl')
    pairs = recovery_pairs(rows)
    if not pairs:
        raise ValueError('No flexible-score recoveries. Run the benchmark CPU summary first.')
    collection = collection_pairs(rows, pairs)
    for pair in collection+pairs:
        for style in ('neutral', 'angry'):
            row = pair[style]
            if row['prompt'] != previous['prompt'] or row['seed'] != 42 or row['max_new_tokens'] != 512:
                raise ValueError('Saved recovery generation settings differ from the benchmark.')
            path = Path(row['audio_file'])
            if digest(path) != row['audio_sha256']:
                raise ValueError(f'Recovery audio missing or changed: {path}')
    plan = dict(protocol=PROTOCOL, benchmark=str(benchmark.absolute()),
        source_config_sha256=digest(benchmark/'run_config.json'),
        source_rescored_sha256=digest(benchmark/'rescored.jsonl'),
        instruct_model=MODEL, instruct_revision=REVISION, base_model=BASE_MODEL, base_revision=BASE_REVISION,
        source_revision=SOURCE_REVISION, tokenizer_model=TOKENIZER_MODEL, tokenizer_revision=TOKENIZER_REVISION,
        prompt=previous['prompt'], seed=42, generation=PARAMS, layers=list(LAYERS), groups=list(GROUPS), strength=1.,
        bank={p['question_id']: previous['bank'][p['question_id']] for p in collection+pairs},
        collection_pairs=collection, pairs=pairs,
        selection='125 seed42 collection pairs chosen from the remaining questions after excluding ALL flexible-score angry-correct/neutral-wrong test questions. Both emotion and checkpoint differences use these SAME 125 collection questions. Sweep uses ONLY the disjoint recovery test questions. Test outcomes are selected; diagnostic only, not overall accuracy.',
        emotion_vector='Equal-question average of mean angry minus mean neutral last-position decoder outputs during their native greedy generations, including prefill and EOS-prediction forwards.',
        audio_vector='Equal-question average of angry minus neutral mean AUDIO token outputs during prefill, at positions selected by the native is_continuous_mask. Excludes text instructions and special delimiter tokens.',
        matched_emotion='Angry minus neutral using the same neutral-Instruct answer-token history; audio positions retain their own offsets.',
        checkpoint_difference='Instruct minus base on identical frozen Instruct-encoded neutral audio/text inputs and identical neutral-Instruct answer-token history. Decoder-checkpoint comparison; no base generation or base steering.',
        intervention='One of 28 shared multimodal decoder layers and one token group at a time. AUDIO: add the audio vector x1 to native masked audio positions during prefill ONLY. ANSWER: add the answer vector x1 to the last position on prefill and every cached answer forward. Six speech-output-branch layers excluded.')
    output.mkdir(parents=True, exist_ok=True)
    freeze(output/'run_config.json', plan)
    print(f'Collect AUDIO + ANSWER on {len(collection)} pairs; steer {len(pairs)} DISJOINT neutral recovery recordings; 28 layers × 2 groups × x1. DIAGNOSTIC ONLY.', flush=True)
    return plan


def atomic_save(path, value):
    import torch
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    return value


def device_copy(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [device_copy(v, device) for v in value]
    return value


def decoder_layers(model):
    layers = model.alm.model.layers
    if len(layers) != 28 or model.alm.config.hidden_size != 3584:
        raise ValueError('Kimi shared multimodal decoder architecture differs from the inspected checkpoints.')
    return layers


@contextmanager
def capture_means(layers):
    """Only the last position predicts the next answer token; never pool audio tokens."""
    sums, counts, handles = {}, Counter(), []
    def make_hook(index):
        def hook(module, inputs, outputs):
            vector = outputs[0][0, -1].detach().float()
            if index not in sums:
                sums[index] = vector.clone()
            else:
                sums[index].add_(vector)
            counts[index] += 1
        return hook
    try:
        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(i)))
        yield sums, counts
    finally:
        for handle in handles:
            handle.remove()


def finish_means(sums, counts):
    import torch
    if set(sums) != set(range(len(sums))) or len(set(counts.values())) != 1:
        raise ValueError('Incomplete per-layer activation capture.')
    return torch.stack([(sums[i]/counts[i]).cpu() for i in range(len(sums))])


@contextmanager
def capture_audio_means(layers, state):
    """Capture only native audio slots in prefill, never instructions or answer tokens."""
    values, handles = {}, []
    def make_hook(index):
        def hook(module, inputs, outputs):
            mask = state.get('mask')
            if index in values or mask is None:
                return
            hidden = outputs[0]
            if mask.shape != hidden.shape[:2] or not mask.any():
                raise ValueError('Native audio mask is empty or differs from prefill positions.')
            values[index] = hidden[0, mask[0].bool()].detach().float().mean(0).cpu()
        return hook
    try:
        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(i)))
        yield values
    finally:
        for handle in handles:
            handle.remove()


def finish_audio(values, count):
    import torch
    if set(values) != set(range(count)):
        raise ValueError('Missing audio-token captures; refusing to substitute answer activations.')
    return torch.stack([values[i] for i in range(count)])


def seed():
    import numpy as np
    import torch
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)


def generate(model, row, prompt):
    from evaluate_kimi_audio import messages
    seed()
    _, response = model.generate(messages(Path(row['audio_file']), prompt), **PARAMS)
    return response


def trace_generation(model, row, prompt):
    """Use the unmodified upstream greedy loop; record actual token inputs, not retokenized prose."""
    calls, first, audio_state = [], {}, {}
    original = model.alm.forward
    def forward(*args, **kwargs):
        audio_state['mask'] = kwargs.get('is_continuous_mask')
        if not calls:
            first.update({k: cpu_copy(kwargs[k]) for k in ('input_ids', 'text_input_ids',
                'whisper_input_feature', 'is_continuous_mask', 'position_ids')})
        calls.append((int(kwargs['input_ids'][0, -1]), int(kwargs['text_input_ids'][0, -1])))
        return original(*args, **kwargs)
    layers = decoder_layers(model)
    with capture_means(layers) as (sums, counts), capture_audio_means(layers, audio_state) as audio, patch.object(model.alm, 'forward', forward):
        response = generate(model, row, prompt)
    if next(iter(counts.values())) != len(calls):
        raise ValueError('Decoder layer and native forward counts differ.')
    return dict(response=response, means=finish_means(sums, counts),
                audio_means=finish_audio(audio, len(layers)), inputs=first, calls=calls)


def encode(model, row, prompt):
    import torch
    from evaluate_kimi_audio import messages
    history = model.prompt_manager.get_prompt(messages(Path(row['audio_file']), prompt), output_type='text')
    audio, text, mask, _, _ = history.to_tensor()
    return dict(input_ids=audio, text_input_ids=text, is_continuous_mask=mask,
                whisper_input_feature=history.continuous_feature,
                position_ids=torch.arange(audio.shape[1]).unsqueeze(0))


def replay_means(model, inputs, calls, include_audio=False):
    """Same answer token history, own audio-prefix offset, KV cache reset for every replay."""
    import torch
    device = next(model.alm.parameters()).device
    initial = {k: device_copy(v, device) for k, v in inputs.items()}
    first_length = initial['input_ids'].shape[1]
    past, state = None, {}
    layers = decoder_layers(model)
    with torch.inference_mode(), capture_means(layers) as (sums, counts), capture_audio_means(layers, state) as audio:
        for step, (audio_token, text_token) in enumerate(calls):
            if step == 0:
                kwargs = initial
            else:
                kwargs = dict(input_ids=torch.tensor([[audio_token]], device=device),
                    text_input_ids=torch.tensor([[text_token]], device=device),
                    whisper_input_feature=None, is_continuous_mask=None,
                    position_ids=torch.tensor([[first_length+step-1]], device=device))
            state['mask'] = kwargs.get('is_continuous_mask')
            result = model.alm.forward(**kwargs, past_key_values=past, return_dict=False)
            past = result[2]
    means = finish_means(sums, counts)
    if include_audio:
        return dict(means=means, audio_means=finish_audio(audio, len(layers)))
    return means


@contextmanager
def steer_layer(layer, direction, strength=1., group='answer', audio_state=None):
    def hook(module, inputs, outputs):
        mask = None
        if group == 'audio':
            mask = audio_state.get('mask') if audio_state is not None else None
            if mask is None:
                return outputs  # Cached answer steps contain no input audio positions.
            if mask.shape != outputs[0].shape[:2] or not mask.any():
                raise ValueError('Cannot steer audio without a valid native audio-position mask.')
        elif group != 'answer':
            raise ValueError(f'Unknown steering group: {group}')
        hidden = outputs[0].clone()
        delta = direction.to(device=hidden.device, dtype=hidden.dtype)*strength
        if group == 'audio':
            hidden[mask.bool()] += delta
        else:
            hidden[:, -1, :] += delta
        return (hidden, *outputs[1:])
    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def load_model(source, base=False):
    from evaluate_kimi_audio import assets, MODEL, REVISION
    model_path, tokenizer_path, KimiAudio = assets(source,
        BASE_MODEL if base else MODEL, BASE_REVISION if base else REVISION)
    from kimia_infer.api.prompt_manager import Glm4Tokenizer
    with patch('kimia_infer.api.prompt_manager.Glm4Tokenizer', side_effect=lambda _: Glm4Tokenizer(tokenizer_path)):
        model = KimiAudio(model_path=model_path, load_detokenizer=False)
    model.alm.eval()
    decoder_layers(model)
    return model


def clear_model():
    import torch
    gc.collect(); torch.cuda.empty_cache()


def cache_file(output, pair, base=False):
    key = hashlib.sha256((pair['question_id']+'\0'+pair['voice']).encode()).hexdigest()[:20]
    return output/'pairs'/f'{key}{"-base" if base else ""}.pt'


def collect(plan, output, model):
    import torch
    for i, pair in enumerate(plan['collection_pairs'], 1):
        path = cache_file(output, pair)
        if path.exists():
            continue
        print(f'Collect [{i}/{len(plan["collection_pairs"])}] {pair["question_id"]}: all 28 layers', flush=True)
        neutral = trace_generation(model, pair['neutral'], plan['prompt'])
        angry = trace_generation(model, pair['angry'], plan['prompt'])
        for style, trace in [('neutral', neutral), ('angry', angry)]:
            if trace['response'] != pair[style]['response']:
                (output/'reproduction-mismatch.json').write_text(json.dumps(dict(question_id=pair['question_id'],
                    style=style, expected=pair[style]['response'], actual=trace['response']), indent=2)+'\n')
                raise ValueError('Native rerun differs from saved Kimi answer. Inspect reproduction-mismatch.json before continuing.')
        matched_angry = replay_means(model, encode(model, pair['angry'], plan['prompt']), neutral['calls'])
        # Exact unchanged-input replay is an implementation check, not another generation.
        replay = replay_means(model, neutral['inputs'], neutral['calls'], include_audio=True)
        replay_error = max((replay[k]-neutral[k]).abs().max().item() for k in ('means', 'audio_means'))
        if any(not torch.allclose(replay[k], neutral[k], rtol=1e-5, atol=1e-5) for k in ('means', 'audio_means')):
            raise ValueError(f'Native neutral/replay activation mismatch: {replay_error}')
        atomic_save(path, dict(question_id=pair['question_id'], voice=pair['voice'], neutral=neutral,
            angry_means=angry['means'], emotion=angry['means']-neutral['means'],
            angry_audio_means=angry['audio_means'], audio_emotion=angry['audio_means']-neutral['audio_means'],
            matched_emotion=matched_angry-neutral['means'], replay_max_abs_error=replay_error))


def collect_base(plan, output, model):
    import torch
    for i, pair in enumerate(plan['collection_pairs'], 1):
        path = cache_file(output, pair, base=True)
        if path.exists():
            continue
        print(f'Base comparison [{i}/{len(plan["collection_pairs"])}] {pair["question_id"]}: fixed neutral history', flush=True)
        data = torch.load(cache_file(output, pair), map_location='cpu', weights_only=True)
        base = replay_means(model, data['neutral']['inputs'], data['neutral']['calls'], include_audio=True)
        atomic_save(path, dict(base_means=base['means'], checkpoint_difference=data['neutral']['means']-base['means'],
            base_audio_means=base['audio_means'], audio_checkpoint_difference=data['neutral']['audio_means']-base['audio_means']))


def mean_vectors(plan, output):
    import torch
    data = [torch.load(cache_file(output, p), map_location='cpu', weights_only=True) for p in plan['collection_pairs']]
    vectors = {name: torch.stack([d[name] for d in data]).mean(0) for name in ('emotion', 'matched_emotion', 'audio_emotion')}
    atomic_save(output/'style-vectors.pt', dict(**vectors, layers=list(LAYERS), protocol=PROTOCOL,
        question_ids=[p['question_id'] for p in plan['collection_pairs']], strength=1.))
    return data, vectors


def cosine_report(plan, output, data, vectors):
    import torch
    from torch.nn.functional import cosine_similarity
    bases = [cache_file(output, p, base=True) for p in plan['collection_pairs']]
    if not all(p.exists() for p in bases):
        return
    rows = [torch.load(p, map_location='cpu', weights_only=True) for p in bases]
    differences = {}
    records = []
    for group in GROUPS:
        audio = group == 'audio'
        key = 'audio_checkpoint_difference' if audio else 'checkpoint_difference'
        checkpoint = torch.stack([d[key] for d in rows]).mean(0)
        neutral = torch.stack([d['neutral']['audio_means' if audio else 'means'] for d in data]).mean(0)
        base = torch.stack([d['base_audio_means' if audio else 'base_means'] for d in rows]).mean(0)
        emotion = vectors['audio_emotion' if audio else 'emotion']
        matched = emotion if audio else vectors['matched_emotion']
        differences[group] = dict(checkpoint_difference=checkpoint, instruct_neutral_mean=neutral, base_neutral_mean=base)
        for layer in LAYERS:
            cos = lambda a, b: float(cosine_similarity(a[layer], b[layer], dim=0))
            records.append(dict(group=group, layer=layer, matched_emotion_vs_checkpoint_cosine=cos(matched, checkpoint),
                native_emotion_vs_checkpoint_cosine=cos(emotion, checkpoint),
                instruct_vs_base_representation_cosine=cos(neutral, base),
                emotion_norm=float(emotion[layer].norm()), matched_emotion_norm=float(matched[layer].norm()),
                checkpoint_difference_norm=float(checkpoint[layer].norm())))
    atomic_save(output/'checkpoint-differences.pt', dict(**differences, layers=list(LAYERS), protocol=PROTOCOL))
    with (output/'layer-cosines.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print('Saved layer-cosines.csv: descriptive checkpoint comparison, not proof of an alignment mechanism.', flush=True)


def condition_name(layer, group):
    return f'neutral-Kimi-L{layer}-{group}-a1'


def generate_steered(model, row, prompt, layer, group, direction):
    state = {}
    original = model.alm.forward
    def forward(*args, **kwargs):
        state['mask'] = kwargs.get('is_continuous_mask')
        return original(*args, **kwargs)
    with patch.object(model.alm, 'forward', forward), steer_layer(
            decoder_layers(model)[layer], direction, group=group, audio_state=state):
        return generate(model, row, prompt)


def sweep(plan, output, model, vectors, max_generations=0):
    results = output/'results.jsonl'
    rows = read_rows(results, repair=True)
    done = {(r['question_id'], r['voice'], r['condition']) for r in rows}
    with results.open('a') as stream:
        for pair in plan['pairs']:
            for style in ('neutral', 'angry'):
                key = pair['question_id'], pair['voice'], style
                if key not in done:
                    stream.write(json.dumps(dict(pair[style], condition=style, reused_from=plan['benchmark']))+'\n')
                    done.add(key)
        stream.flush()
        completed = 0
        for layer, group in [(i, g) for i in LAYERS for g in GROUPS]:
            condition = condition_name(layer, group)
            for pair in plan['pairs']:
                key = pair['question_id'], pair['voice'], condition
                if key in done:
                    continue
                print(f'Steer L{layer} {group} x1 [{completed+1}] {pair["question_id"]}', flush=True)
                started = time.monotonic()
                response = generate_steered(model, pair['neutral'], plan['prompt'], layer, group,
                    vectors['audio_emotion' if group == 'audio' else 'emotion'][layer])
                row = dict(pair['neutral'], condition=condition, response=response,
                    generation_seconds=time.monotonic()-started, reused_from=None,
                    interventions=[dict(layer=layer, group=group, strength=1.)],
                    diagnostic=PROTOCOL)
                row.pop('harness', None)
                row.pop('strict-format-extractable', None)
                stream.write(json.dumps(row)+'\n'); stream.flush()
                done.add(key); completed += 1
                if max_generations and completed >= max_generations:
                    return


def report(output):
    from score_bbh_harness import HarnessScorer
    from evaluate_bbh_audio_models import strict_format_bits
    plan = json.loads((output/'run_config.json').read_text())
    rows = read_rows(output/'results.jsonl')
    if not rows:
        collected = sum(cache_file(output, pair).exists() for pair in plan['collection_pairs'])
        compared = sum(cache_file(output, pair, base=True).exists() for pair in plan['collection_pairs'])
        print(f'Collection: {collected}/{len(plan["collection_pairs"])} pairs; base comparison: {compared}/{len(plan["collection_pairs"])}. No sweep answers yet.')
        return
    scorer = HarnessScorer(output/'harness/upstream')
    docs = [dict(input=plan['bank'][r['question_id']]['original_input'],
                 target=plan['bank'][r['question_id']]['original_target']) for r in rows]
    scores = scorer.score([r['response'] for r in rows], docs)
    formats = strict_format_bits(scorer, [r['response'] for r in rows], docs)
    records = [dict(row, harness={k: v[i] for k, v in scores.items()},
                    **{'strict-format-extractable': formats[i]}) for i, row in enumerate(rows)]
    print('\nRECOVERY-SELECTED LAYER DIAGNOSTIC. 125 collection questions; disjoint recovery-only test questions. No population-level p-values.')
    print('Condition                       N  Flexible   Strict   Format')
    summaries = []
    for condition in ['neutral', 'angry', *[condition_name(i, g) for i in LAYERS for g in GROUPS]]:
        group = [r for r in records if r['condition'] == condition]
        if not group:
            continue
        counts = {k: sum(r['harness'][k]['correct'] for r in group) for k in scores}
        counts['strict-format-extractable'] = sum(r['strict-format-extractable'] for r in group)
        summaries.append(dict(condition=condition, n=len(group), correct=counts))
        print(f'{condition:29s} {len(group):3d}  {counts["flexible-extract"]/len(group):7.1%}'
              f'  {counts["strict-match"]/len(group):7.1%}  {counts["strict-format-extractable"]/len(group):7.1%}')
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (output/'summary.json').write_text(json.dumps(dict(protocol=PROTOCOL, selection=plan['selection'],
        scorer=scorer.provenance, conditions=summaries, note='125 collection questions and recovery-only test questions are disjoint. Test questions were selected by outcomes; the sweep does not estimate overall accuracy gain or establish a validated best layer.'), indent=2)+'\n')


def run(args, plan):
    (args.output/'pairs').mkdir(exist_ok=True)
    if args.mode in ('all', 'collect') and any(not cache_file(args.output, p).exists() for p in plan['collection_pairs']):
        model = load_model(args.model_source)
        collect(plan, args.output, model)
        del model; clear_model()
    data, vectors = mean_vectors(plan, args.output)
    if args.mode in ('all', 'compare') and any(not cache_file(args.output, p, True).exists() for p in plan['collection_pairs']):
        print('Base comparison downloads another checkpoint (~17 GB). Existing weights/data are never removed.', flush=True)
        model = load_model(args.model_source, base=True)
        collect_base(plan, args.output, model)
        del model; clear_model()
    cosine_report(plan, args.output, data, vectors)
    if args.mode in ('all', 'sweep'):
        done = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(args.output/'results.jsonl')}
        pending = any((p['question_id'], p['voice'], condition_name(i, g)) not in done
                      for p in plan['pairs'] for i in LAYERS for g in GROUPS)
        if pending:
            model = load_model(args.model_source)
            sweep(plan, args.output, model, vectors, args.max_generations)
            del model; clear_model()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', type=Path, default=Path('outputs/bbh-three-regenerated-models/kimi'))
    p.add_argument('--output', type=Path, default=Path('outputs/kimi-bbh3-collect125-audio-answer-steer30'))
    p.add_argument('--model-source', type=Path, default=Path('external/Kimi-Audio'))
    p.add_argument('--mode', choices=['all', 'prepare', 'collect', 'compare', 'sweep', 'summary'], default='all')
    p.add_argument('--max-generations', type=int, default=0, help='Optional resumable sweep smoke limit; 0 means all 28 layers for BOTH audio and answer positions.')
    args = p.parse_args()
    if args.max_generations < 0:
        p.error('--max-generations must be nonnegative')
    if args.mode == 'summary':
        report(args.output); return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(args.benchmark, args.output)
        if args.mode == 'prepare':
            return
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            run(args, plan)


if __name__ == '__main__':
    main()
