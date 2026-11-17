"""Phi-4 GSM8K: fixed 400 collection /400 test; neutral L17 answer -0.5/-1."""
import argparse
from contextlib import contextmanager
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

from bbh_candidate_adapters import MODELS, checkpoint, load_audio, phi_prompt
from collect_phi4_reasoning_emotion import decoder_layers
import evaluate_gsm_candidates as gsm
from kimi_bbh_layer_diagnostic import atomic_save, digest, freeze
from steer import read_rows

PROTOCOL = 'phi4-gsm8k-random400-400-L17-reverse-native-mean-v1'
LAYER = 17  # Zero based: eighteenth decoder block.
STRENGTHS = {'neutral': 0., 'neutral-L17-answer-a-0.5': -.5, 'neutral-L17-answer-a-1': -1.}


def prepare(args):
    path = args.output/'run_config.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if (plan['protocol'] != PROTOCOL or plan['source'] != str(args.source.absolute())
                or plan['prompt'] != gsm.ORIGINAL or plan['recipe']['revision'] != MODELS['phi4']['revision']):
            raise ValueError('Frozen reverse-steering settings changed; use another output folder.')
        print('Reusing fixed400 collection /400 disjoint test questions.', flush=True)
        return plan
    parent = gsm.make_plan(args.source, 'phi4', 'original')
    ordered = gsm.questions()  # Canonical1269 question bank, shuffled with seed42.
    if len(ordered) < 800: raise ValueError('Need800 canonical GSM8K questions.')
    train, test = ordered[:400], ordered[400:800]
    records = {(r['question_id'], r['style']): r for r in parent['recordings']}
    collection = []
    for q in train:
        n, a = records[q['question_id'], 'neutral'], records[q['question_id'], 'angry']
        if n['voice'] != a['voice']: raise ValueError('Collection voices differ.')
        collection.append(dict(question_id=q['question_id'], voice=n['voice'], neutral=n, angry=a))
    evaluation = [records[q['question_id'], 'neutral'] for q in test]
    if {p['question_id'] for p in collection} & {r['question_id'] for r in evaluation}:
        raise ValueError('Collection and test overlap.')
    plan = dict(protocol=PROTOCOL, source=str(args.source.absolute()), recipe=parent['recipe'],
                prompt=gsm.ORIGINAL, seed=42, max_new_tokens=512, layer=LAYER, group='answer', strengths=STRENGTHS,
                collection=collection, test=evaluation, bank={q['question_id']: q for q in test},
                collection_bank={q['question_id']: q for q in train}, question_file_sha256=digest(gsm.QUESTION_FILE),
                vector_definition='Mean of400 per-question (angry minus neutral) L17 means. Each mean equally averages '
                    'last-position answer-prediction states across every native greedy generation forward, '
                    'including initial prefill. Own generated histories; no teacher forcing or gold answer input. '
                    'Raw magnitude retained, no normalization. Not the earlier initial-forward-only cosine vector.',
                selection='Canonical1269 bank shuffled seed42; first400 collection, next400 test. No outcome, '
                    'format, emotion-classifier or duration filtering. Prior model outcomes were inspected; exploratory follow-up.',
                interventions='Subtract angry-neutral from neutral audio at last prediction position of L17 on every '
                    'native forward. Strengths0,-0.5,-1; same prompt/seed/decoder across conditions.',
                scoring='Installed upstream GSM8K task filters/metric; numeric equivalence additional. '
                    'Audio-adapted inference, not full official text-harness inference. Tests/intervals unadjusted.')
    freeze(path, plan)
    print('Frozen400 neutral/angry collection pairs;400 disjoint neutral tests ×3 conditions. '
          '800 collection +1200 evaluation generations; existing full audio reused.', flush=True)
    return plan


def seed():
    import numpy as np
    import torch
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)


@contextmanager
def answer_hook(layer, direction=None, strength=0., observe=False):
    """Touch only the last prediction state, preserving output tuple/KV cache."""
    import torch
    state = dict(count=0, total=None)
    if not observe and strength == 0:
        yield state
        return
    if strength and (direction is None or direction.ndim != 1 or not torch.isfinite(direction).all()):
        raise ValueError('Invalid steering direction.')
    def hook(module, inputs, outputs):
        h = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if h.ndim != 3 or h.shape[0] != 1: raise ValueError('Expected one native answer sequence.')
        state['count'] += 1
        if observe:
            value = h[0, -1].detach().float().cpu().clone()
            state['total'] = value if state['total'] is None else state['total']+value
        if not strength: return outputs
        if h.shape[-1] != direction.numel(): raise ValueError('Vector hidden size differs from native decoder.')
        changed = h.clone()
        changed[:, -1, :] += strength*direction.to(device=h.device, dtype=h.dtype)
        return (changed, *outputs[1:]) if isinstance(outputs, tuple) else ([changed, *outputs[1:]] if isinstance(outputs, list) else changed)
    handle = layer.register_forward_hook(hook)
    try: yield state
    finally: handle.remove()


class NativePhi:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
        if version('transformers') != MODELS['phi4']['transformers']:
            raise ValueError('Use existing .venv-bbh-phi4 environment.')
        path = checkpoint('phi4')
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map='cuda:0', _attn_implementation='sdpa').eval()
        self.config = GenerationConfig.from_pretrained(path)
        self.layer = decoder_layers(self.model)[LAYER]

    def generate(self, row, prompt, direction=None, strength=0., observe=False):
        import torch
        if digest(row['audio_file']) != row['audio_sha256']: raise ValueError('Audio changed since selection.')
        seed()
        inputs = self.processor(text=phi_prompt(prompt), audios=[(load_audio(row['audio_file']), 16000)],
                                return_tensors='pt').to(self.model.device, dtype=torch.bfloat16)
        with torch.inference_mode(), answer_hook(self.layer, direction, strength, observe) as state:
            ids = self.model.generate(**inputs, generation_config=self.config, max_new_tokens=512,
                do_sample=False, num_beams=1, repetition_penalty=1., use_cache=True)
        suffix = ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        result = dict(response=response, response_tokens=int(suffix.shape[1]), hook_forwards=state['count'])
        if observe:
            if not state['count']: raise ValueError('No native decoder forwards captured.')
            result['mean'] = state['total']/state['count']
        return result


def validate_pair(saved, pair, config_hash):
    import torch
    if (saved['protocol'] != PROTOCOL or saved['run_config_sha256'] != config_hash
            or saved['question_id'] != pair['question_id'] or saved['voice'] != pair['voice']
            or saved['audio_sha256'] != {s: pair[s]['audio_sha256'] for s in ('neutral', 'angry')}):
        raise ValueError('Collection cache differs from frozen experiment.')
    if saved['answer_difference'].shape != (3072,) or not torch.isfinite(saved['answer_difference']).all():
        raise ValueError('Invalid collected direction.')
    for trace in saved['traces'].values():
        if trace['hook_forwards'] <= 0 or trace['mean'].shape != (3072,) or not torch.isfinite(trace['mean']).all():
            raise ValueError('Invalid cached native generation mean/count.')
    if not torch.equal(saved['answer_difference'], saved['traces']['angry']['mean']-saved['traces']['neutral']['mean']):
        raise ValueError('Cached difference sign/value disagrees with native means.')


def collect(plan, output, native, config_hash):
    import torch
    folder = output/'pairs'; folder.mkdir(exist_ok=True)
    differences = []
    for i, pair in enumerate(plan['collection']):
        path = folder/f'{i:04d}.pt'
        if path.exists(): saved = torch.load(path, map_location='cpu', weights_only=True)
        else:
            print(f'Collect [{i+1}/400] {pair["question_id"]}: neutral + angry, native L17 means.', flush=True)
            traces = {s: native.generate(pair[s], plan['prompt'], observe=True) for s in ('neutral', 'angry')}
            saved = dict(protocol=PROTOCOL, run_config_sha256=config_hash, question_id=pair['question_id'], voice=pair['voice'],
                         audio_sha256={s: pair[s]['audio_sha256'] for s in traces},
                         traces=traces, answer_difference=traces['angry']['mean']-traces['neutral']['mean'])
            validate_pair(saved, pair, config_hash); atomic_save(path, saved)
        validate_pair(saved, pair, config_hash); differences.append(saved['answer_difference'])
    vector = dict(protocol=PROTOCOL, run_config_sha256=config_hash, model=plan['recipe']['model'],
                  revision=plan['recipe']['revision'], layer=LAYER, group='answer', definition=plan['vector_definition'],
                  collection_question_ids=[p['question_id'] for p in plan['collection']],
                  angry_minus_neutral=torch.stack(differences).mean(0))
    path = output/'style-vectors.pt'
    if path.exists():
        old = torch.load(path, map_location='cpu', weights_only=True)
        if (old['run_config_sha256'] != config_hash or not torch.equal(old['angry_minus_neutral'], vector['angry_minus_neutral'])):
            raise ValueError('Existing vector differs from completed collection.')
    else: atomic_save(path, vector)
    return vector['angry_minus_neutral']


def completed(plan, output, config_hash, repair=False):
    rows = read_rows(output/'results.jsonl', repair=repair); done = {}
    tasks = {(r['question_id'], r['voice']): r for r in plan['test']}
    for r in rows:
        key = r['question_id'], r['voice']; c = r['condition']
        if (key not in tasks or c not in STRENGTHS or r['protocol'] != PROTOCOL or r['run_config_sha256'] != config_hash
                or r['prompt'] != plan['prompt'] or r['audio_sha256'] != tasks[key]['audio_sha256']
                or r['reference_answer'] != plan['bank'][r['question_id']]['answer']
                or r['seed'] != 42 or r['max_new_tokens'] != 512 or r['strength'] != STRENGTHS[c]
                or r['model'] != plan['recipe']['model'] or r['revision'] != plan['recipe']['revision']):
            raise ValueError('Test answer differs from frozen reverse-steering protocol.')
        full_key = (*key, c)
        if full_key in done: raise ValueError('Duplicate test result; inspect before resuming.')
        done[full_key] = r
    return rows, done


def evaluate(plan, output, native, vector, config_hash):
    _, done = completed(plan, output, config_hash, repair=True)
    vector_hash = digest(output/'style-vectors.pt')
    jobs = [(r, c, a) for r in plan['test'] for c, a in STRENGTHS.items() if (r['question_id'], r['voice'], c) not in done]
    for i, (row, condition, strength) in enumerate(jobs, 1):
        print(f'Evaluate [{i}/{len(jobs)}] {row["question_id"]} {condition}', flush=True)
        start = time.monotonic()
        trace = native.generate(row, plan['prompt'], vector if strength else None, strength)
        saved = dict(row, **trace, condition=condition, protocol=PROTOCOL, run_config_sha256=config_hash,
                     prompt=plan['prompt'], seed=42, max_new_tokens=512, model=plan['recipe']['model'],
                     revision=plan['recipe']['revision'], strength=strength, layer=LAYER, group='answer',
                     vector_sha256=vector_hash if strength else None, generation_seconds=time.monotonic()-start,
                     interventions=[dict(layer=LAYER, group='answer', strength=strength)] if strength else [])
        with (output/'results.jsonl').open('a') as stream: stream.write(json.dumps(saved)+'\n'); stream.flush()
        print(trace['response'], flush=True)


def report(output):
    from rescore_audio_steering import HarnessTaskScorer, numeric_equal, paired
    path = output/'run_config.json'
    if not path.exists(): print('Phi reverse steering: not prepared.'); return
    plan = json.loads(path.read_text()); rows, _ = completed(plan, output, digest(path))
    print(f'Phi-4 reverse L17: {len(rows)}/1200 test answers; {len(list((output/"pairs").glob("*.pt")))}/400 collection pairs.', flush=True)
    if not rows: return
    scorer = HarnessTaskScorer('gsm8k', output/'harness/upstream')
    docs = [dict(question=plan['bank'][r['question_id']]['question'], answer=r['reference_answer']) for r in rows]
    values = scorer.score([r['response'] for r in rows], docs)
    scored = [dict(r, scores={**{m: v[i]['correct'] for m, v in values.items()},
        'flexible-numeric': numeric_equal(values['flexible-extract'][i]['prediction'], r['reference_answer']),
        'format-compliant': values['strict-match'][i]['prediction'] != '[invalid]'}) for i, r in enumerate(rows)]
    groups = {c: {(r['question_id'], r['voice']): r for r in scored if r['condition'] == c} for c in STRENGTHS}
    keys = sorted(set.intersection(*(set(g) for g in groups.values()))); n = len(keys)
    print(f'{n}/400 identical questions completed in all THREE conditions.', flush=True)
    accuracy, comparisons = [], []
    metrics = ('flexible-numeric', 'flexible-extract', 'strict-match', 'format-compliant')
    if n:
        print('Condition                        N   Numeric  Flexible   Strict   Format  Mean tokens  Median tokens')
        for c in STRENGTHS:
            counts = {m: sum(groups[c][k]['scores'][m] for k in keys) for m in metrics}
            lengths = [groups[c][k]['response_tokens'] for k in keys]
            accuracy.append(dict(condition=c, n=n, counts=counts, mean_tokens=statistics.mean(lengths), median_tokens=statistics.median(lengths)))
            print(f'{c:30} {n:4d} '+ ' '.join(f'{counts[m]/n:8.1%}' for m in metrics)
                  +f' {statistics.mean(lengths):12.1f} {statistics.median(lengths):14.1f}')
        for c in list(STRENGTHS)[1:]:
            for m in metrics:
                result = paired([(groups['neutral'][k]['scores'][m], groups[c][k]['scores'][m]) for k in keys])
                comparisons.append(dict(condition=c, metric=m, **result))
                print(f'{c} / {m}: gain={result["gain_pp"]:+.2f} pp; W/L={result["wins"]}/{result["losses"]}; '
                      f'p={result["mcnemar_exact_p"]:.4g}; CI={result["ci95_gain_pp"]}', flush=True)
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    (output/'summary.json').write_text(json.dumps(dict(protocol=PROTOCOL, matched_questions=n, accuracy=accuracy,
        comparisons=comparisons, scorer=scorer.provenance, note=plan['scoring']), indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('prepare', 'evaluate', 'summary'), default='evaluate')
    p.add_argument('--source', type=Path, default=Path('outputs/azure-kimi-gsm8k1269'))
    p.add_argument('--output', type=Path, default=Path('outputs/phi4-gsm-reverse400-400-L17'))
    args = p.parse_args()
    if args.mode == 'summary': report(args.output); return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB); plan = prepare(args)
        if args.mode == 'prepare': return
        config_hash = digest(args.output/'run_config.json')
        _, done = completed(plan, args.output, config_hash, repair=True)
        if len(done) == 1200: print('All1200 test answers retained; no model loading.'); return
        with Path('outputs/jenny-task-vectors.lock').open('w') as gpu:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(gpu, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            native = NativePhi()
            import torch
            (args.output/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__, transformers=version('transformers'),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), collector_sha256=digest(__file__),
                generation='native Phi greedy, BF16/SDPA, KV cache, max512, seed42, repetition_penalty1'), indent=2)+'\n')
            vector = collect(plan, args.output, native, config_hash)
            evaluate(plan, args.output, native, vector, config_hash)


if __name__ == '__main__': main()
