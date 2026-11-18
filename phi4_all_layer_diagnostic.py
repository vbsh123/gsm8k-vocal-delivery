"""Phi all-layer -1 diagnostic: collect400; recoveries selected only from remaining869."""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import json
from pathlib import Path
import statistics
import time

import evaluate_gsm_candidates as gsm
import evaluate_bbh_candidates as candidates
import steer_phi4_gsm_reverse as engine
from kimi_bbh_layer_diagnostic import atomic_save, digest, freeze
from pack_experiment_backup import snapshot
from steer import read_rows

PROTOCOL = 'phi4-gsm400-all32-test869-recovery-minus1-v1'
SHAPE = (32, 2, 3072)
GROUPS = ('answer', 'audio')
TRAIN = 400
TOTAL = 1269


def prepare(args):
    config = args.output/'run_config.json'
    if config.exists():
        plan = json.loads(config.read_text())
        if plan['protocol'] != PROTOCOL or plan['source_collection'] != str(args.collection.absolute()):
            raise ValueError('Frozen all-layer settings differ.')
        if digest(Path(plan['source_config'])) != plan['source_config_sha256']: raise ValueError('Source config changed.')
        return plan
    old_path = args.collection/'run_config.json'; old = json.loads(old_path.read_text()); old_hash = digest(old_path)
    if (len(old['collection']) != TRAIN or old['prompt'] != gsm.ORIGINAL or old['seed'] != 42
            or old['max_new_tokens'] != 512 or old['recipe']['revision'] != gsm.MODELS['phi4']['revision']):
        raise ValueError('Need original completed400-pair Phi collection with original prompt.')
    questions = gsm.questions()
    if len(questions) != TOTAL or [p['question_id'] for p in old['collection']] != [q['question_id'] for q in questions[:TRAIN]]:
        raise ValueError('Collection differs from canonical first400.')
    source = gsm.make_plan(Path(old['source']), 'phi4', 'original')
    if source['recipe'] != old['recipe']: raise ValueError('Native recipe differs.')
    test_ids = {q['question_id'] for q in questions[TRAIN:]}
    records = [r for r in source['recordings'] if r['question_id'] in test_ids]
    if len(records) != 2*(TOTAL-TRAIN): raise ValueError('Need both full neutral/angry audio references on remaining869.')
    receipts = []; original = []
    for i, pair in enumerate(old['collection']):
        path = args.collection/'pairs'/f'{i:04d}.pt'
        import torch
        saved = torch.load(path, map_location='cpu', weights_only=True)
        engine.validate_pair(saved, pair, old_hash, old['protocol'])
        receipts.append(dict(path=str(path.absolute()), sha256=digest(path)))
        for style in ('neutral', 'angry'):
            trace = saved['traces'][style]
            original.append(dict(question_id=pair['question_id'], voice=pair['voice'], style=style,
                response=trace['response'], response_tokens=trace['response_tokens']))
    plan = dict(protocol=PROTOCOL, evaluation_kind='all-layer-diagnostic', source_collection=str(args.collection.absolute()),
        source_config=str(old_path.absolute()), source_config_sha256=old_hash, collection_receipts=receipts,
        source_layer=old['layer'], recipe=old['recipe'], prompt=old['prompt'], seed=42, max_new_tokens=512,
        collection=old['collection'], recordings=records, bank={q['question_id']: q for q in questions},
        layers=list(range(SHAPE[0])), groups=list(GROUPS), strength=-1.,
        source_neutral_run=str(args.neutral_run.absolute()), source_delivery_run=str(args.delivery_run.absolute()),
        vector_definition='Per-layer/group raw400-question mean of angry-minus-neutral. Answer means average '
            'last prediction states over ALL native generation forwards, including prefill; audio means use '
            'the initial native audio positions only. No normalization, no teacher forcing, own generated histories.',
        intervention='One layer and one group at a time, strength-1. Answer positions on every native forward; '
            'audio positions on initial prefill only, then native KV cache used unchanged.',
        selection='Collect on first400 canonical seed42 questions. Evaluate neutral/angry references on ALL remaining869, '
            'then select neutral-numeric-incorrect/angry-numeric-correct cases. No collection/test overlap. '
            'Outcome-selected recovery/layer-search diagnostic, NOT unbiased benchmark accuracy. No p-values on selected cases.',
        scoring='Installed upstream GSM8K strict/flexible filters and exact-match; numeric equivalence additional. '
            'Audio-adapted native generations, not official text-harness inference.', gpu_budget_gib=20.)
    freeze(config, plan)
    (args.output/'collection-source-responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in original))
    print(f'Frozen{TRAIN} collection pairs, {TOTAL-TRAIN} disjoint test references, all32 layers ×answer/audio ×-1.', flush=True)
    return plan


def expected(plan, row):
    return dict(protocol=PROTOCOL, question_id=row['question_id'], voice=row['voice'], style=row['style'],
        audio_sha256=row['audio_sha256'], reference_answer=row['reference_answer'], prompt=plan['prompt'],
        seed=42, max_new_tokens=512, model=plan['recipe']['model'], revision=plan['recipe']['revision'], interventions=[])


def references(plan, output):
    tasks = {(r['question_id'], r['voice'], r['style']): r for r in plan['recordings']}; done = {}
    config_hash = digest(output/'run_config.json')
    for row in read_rows(output/'references.jsonl'):
        key = row['question_id'], row['voice'], row['style']
        if key not in tasks or key in done or row.get('run_config_sha256') != config_hash:
            raise ValueError('Unknown/duplicate reference or changed config.')
        if any(row.get(k) != v for k, v in expected(plan, tasks[key]).items()): raise ValueError('Reference metadata differs.')
        done[key] = row
    return tasks, done


def reuse_references(plan, output):
    tasks, done = references(plan, output); config_hash = digest(output/'run_config.json'); count = 0
    for root in (Path(plan['source_neutral_run']), Path(plan['source_delivery_run'])):
        config = root/'run_config.json'; result = root/'results.jsonl'
        if not config.exists() or not result.exists(): continue
        old = json.loads(config.read_text()); source_hash = digest(config)
        if old['prompt'] != plan['prompt'] or old['recipe'] != plan['recipe'] or old['seed'] != 42 or old['max_new_tokens'] != 512:
            raise ValueError('Reference source model/prompt/decoder differs.')
        old_tasks = {(r['question_id'], r['voice'], r['style']): r for r in old.get('recordings', old.get('test', []))}
        data, _ = snapshot(result)
        with (output/'references.jsonl').open('a') as stream:
            for line in data.splitlines():
                if not line.strip(): continue
                row = json.loads(line); style = row.get('condition', row.get('style'))
                if style not in ('neutral', 'angry'): continue
                key = row['question_id'], row['voice'], style
                if key not in tasks or key in done: continue
                task = tasks[key]
                if (key not in old_tasks or row['audio_sha256'] != old_tasks[key]['audio_sha256']
                        or row['prompt'] != plan['prompt'] or row['seed'] != 42 or row['max_new_tokens'] != 512
                        or row['model'] != plan['recipe']['model'] or row['revision'] != plan['recipe']['revision']
                        or row['interventions'] or row.get('strength', 0) != 0):
                    raise ValueError('Reference source answer not native/unsteered/matched.')
                if 'run_config_sha256' in row and row['run_config_sha256'] != source_hash:
                    raise ValueError('Source reference config hash differs.')
                if row['audio_sha256'] != task['audio_sha256'] or str(row['reference_answer']) != str(task['reference_answer']):
                    continue
                copied = dict(expected(plan, task), run_config_sha256=config_hash,
                    response=row['response'], response_tokens=row['response_tokens'],
                    reused_evidence=dict(folder=str(root.absolute()), run_config_sha256=source_hash))
                stream.write(json.dumps(copied)+'\n'); done[key] = copied; count += 1
    print(f'Reused{count} unsteered reference answers; {len(done)}/{len(tasks)} references available.', flush=True)


@contextmanager
def hooks(layers, audio_mask, collect=False, target=None, group='answer', direction=None, strength=-1.):
    import torch
    totals = {}; counts = {}; audio = {}; handles = []
    def make(index):
        def hook(module, inputs, result):
            h = result[0] if isinstance(result, (tuple, list)) else result
            if h.ndim != 3 or h.shape[0] != 1: raise ValueError('Expected single native decoder sequence.')
            first = counts.get(index, 0) == 0; counts[index] = counts.get(index, 0)+1
            if first and h.shape[1] != audio_mask.numel(): raise ValueError('Initial native audio mask/input length differs.')
            if collect:
                value = h[0, -1].detach().float()
                totals[index] = value.clone() if index not in totals else totals[index]+value
                if first: audio[index] = h[0, audio_mask].detach().float().mean(0)
                return result
            if group == 'audio' and not first: return result
            if group not in GROUPS or direction is None or direction.shape != (h.shape[-1],): raise ValueError('Invalid intervention group/width.')
            changed = h.clone(); shift = strength*direction.to(device=h.device, dtype=h.dtype)
            if group == 'answer': changed[:, -1] += shift
            else: changed[:, audio_mask] += shift
            return (changed, *result[1:]) if isinstance(result, tuple) else ([changed, *result[1:]] if isinstance(result, list) else changed)
        return hook
    try:
        for i in range(len(layers)) if collect else (target,): handles.append(layers[i].register_forward_hook(make(i)))
        yield totals, counts, audio
    finally:
        for handle in handles: handle.remove()


def generate(native, row, prompt, collect=False, target=None, group='answer', direction=None):
    import torch
    if digest(row['audio_file']) != row['audio_sha256']: raise ValueError('Audio changed.')
    engine.seed(); inputs = native.processor(text=engine.phi_prompt(prompt), audios=[(engine.load_audio(row['audio_file']), 16000)],
        return_tensors='pt').to(native.model.device, dtype=torch.bfloat16)
    ids = inputs['input_ids'][0]; audio_mask = (ids == 200011)|(ids <= -10000)
    if not audio_mask.any() or int(audio_mask.sum()) != int(inputs['audio_embed_sizes'].sum()) or audio_mask[-1]:
        raise ValueError('Invalid native audio/answer position mask.')
    attention = inputs.get('attention_mask')
    if attention is not None and (not attention[0, -1].bool() or not attention[0, audio_mask].bool().all()):
        raise ValueError('Native audio/answer positions overlap padding.')
    started = time.monotonic()
    if not collect and target is None:
        with torch.inference_mode(): result = native.model.generate(**inputs, generation_config=native.config,
            max_new_tokens=512, do_sample=False, num_beams=1, repetition_penalty=1., use_cache=True)
        totals, counts, audio = {}, {}, {}
    else:
        with torch.inference_mode(), hooks(native.layers, audio_mask, collect, target, group, direction) as (totals, counts, audio):
            result = native.model.generate(**inputs, generation_config=native.config, max_new_tokens=512,
                do_sample=False, num_beams=1, repetition_penalty=1., use_cache=True)
    suffix = result[:, inputs['input_ids'].shape[1]:]
    trace = dict(response=native.processor.batch_decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0],
        response_tokens=int(suffix.shape[1]), generation_seconds=time.monotonic()-started)
    if collect:
        if len(totals) != len(native.layers) or len(set(counts.values())) != 1: raise ValueError('Missing/unequal layer capture counts.')
        trace.update(means=torch.stack([torch.stack((totals[i]/counts[i], audio[i])).cpu() for i in range(len(native.layers))]),
                     hook_forwards=counts[0], audio_tokens=int(audio_mask.sum()))
    return trace


def validate_capture(saved, pair, config_hash):
    import torch
    if (saved['protocol'] != PROTOCOL or saved['run_config_sha256'] != config_hash
            or saved['question_id'] != pair['question_id'] or saved['voice'] != pair['voice']
            or saved['audio_sha256'] != {s: pair[s]['audio_sha256'] for s in ('neutral', 'angry')}):
        raise ValueError('All-layer pair cache identity differs.')
    for trace in saved['traces'].values():
        if trace['means'].shape != SHAPE or not torch.isfinite(trace['means']).all() or trace['hook_forwards'] <= 0:
            raise ValueError('Invalid all-layer collection means.')


def collect_vectors(plan, output, native):
    import torch
    folder = output/'pairs'; folder.mkdir(exist_ok=True); differences = []; config_hash = digest(output/'run_config.json')
    for i, pair in enumerate(plan['collection']):
        receipt = plan['collection_receipts'][i]; source_path = Path(receipt['path'])
        if digest(source_path) != receipt['sha256']: raise ValueError('Original collection cache changed.')
        original = torch.load(source_path, map_location='cpu', weights_only=True)
        path = folder/f'{i:04d}.pt'
        if path.exists(): saved = torch.load(path, map_location='cpu', weights_only=True)
        else:
            print(f'Collect [{i+1}/{len(plan["collection"])}] {pair["question_id"]}: all32 answer/audio groups.', flush=True)
            traces = {s: generate(native, pair[s], plan['prompt'], collect=True) for s in ('neutral', 'angry')}
            saved = dict(protocol=PROTOCOL, run_config_sha256=config_hash, question_id=pair['question_id'], voice=pair['voice'],
                audio_sha256={s: pair[s]['audio_sha256'] for s in ('neutral', 'angry')}, traces=traces)
            validate_capture(saved, pair, config_hash); atomic_save(path, saved)
        validate_capture(saved, pair, config_hash)
        for s in ('neutral', 'angry'):
            if saved['traces'][s]['response'] != original['traces'][s]['response']:
                raise ValueError('Replayed collection answer differs from original native generation; cache retained for inspection.')
        differences.append(saved['traces']['angry']['means']-saved['traces']['neutral']['means'])
    vector = dict(protocol=PROTOCOL, run_config_sha256=config_hash, recipe=plan['recipe'], groups=list(GROUPS),
        collection_question_ids=[p['question_id'] for p in plan['collection']], definition=plan['vector_definition'],
        angry_minus_neutral=torch.stack(differences).mean(0))
    path = output/'style-vectors.pt'
    if path.exists():
        old = torch.load(path, map_location='cpu', weights_only=True)
        if old['run_config_sha256'] != config_hash or not torch.equal(old['angry_minus_neutral'], vector['angry_minus_neutral']):
            raise ValueError('Frozen all-layer vector differs.')
    else: atomic_save(path, vector)


def score_rows(plan, rows, output):
    from rescore_audio_steering import HarnessTaskScorer, numeric_equal
    scorer = HarnessTaskScorer('gsm8k', output/'harness/upstream')
    if not rows: return [], scorer.provenance
    docs = [dict(question=plan['bank'][r['question_id']]['question'], answer=r['reference_answer']) for r in rows]
    values = scorer.score([r['response'] for r in rows], docs)
    return [dict(r, scores={**{m: v[i]['correct'] for m, v in values.items()},
        'flexible-numeric': numeric_equal(values['flexible-extract'][i]['prediction'], r['reference_answer']),
        'format-compliant': values['strict-match'][i]['prediction'] != '[invalid]'}) for i, r in enumerate(rows)], scorer.provenance


def select_challenge(plan, output):
    tasks, done = references(plan, output)
    if len(done) != len(tasks): raise ValueError('Finish ALL remaining869 neutral/angry references before selecting challenge.')
    rows, provenance = score_rows(plan, list(done.values()), output)
    groups = {s: {(r['question_id'], r['voice']): r for r in rows if r['style']==s} for s in ('neutral', 'angry')}
    keys = sorted(set(groups['neutral'])&set(groups['angry']))
    cases = [dict(question_id=q, voice=v) for q, v in keys if not groups['neutral'][q, v]['scores']['flexible-numeric']
             and groups['angry'][q, v]['scores']['flexible-numeric']]
    train_ids = {p['question_id'] for p in plan['collection']}
    if train_ids&{c['question_id'] for c in cases}: raise ValueError('Challenge overlaps vector collection.')
    challenge = dict(protocol=PROTOCOL, run_config_sha256=digest(output/'run_config.json'),
        references_sha256=digest(output/'references.jsonl'), primary_metric='flexible-numeric',
        available_test_questions=len(keys), cases=cases, selection=plan['selection'])
    freeze(output/'challenge.json', challenge)
    (output/'reference-rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (output/'reference-scoring-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'Selected{len(cases)} angry-correct/neutral-wrong cases from{len(keys)} disjoint test questions.', flush=True)
    return challenge


def completed(plan, output, challenge):
    config_hash = digest(output/'run_config.json'); challenge_hash = digest(output/'challenge.json')
    cases = {(c['question_id'], c['voice']) for c in challenge['cases']}
    tasks = {(r['question_id'], r['voice']): r for r in plan['recordings'] if r['style']=='neutral'}
    vector_hash = digest(output/'style-vectors.pt'); rows = read_rows(output/'results.jsonl'); done = set()
    for row in rows:
        key = row['question_id'], row['voice']; full = (*key, row['layer'], row['group'])
        if (key not in cases or full in done or row['layer'] not in plan['layers'] or row['group'] not in GROUPS
                or row['run_config_sha256'] != config_hash or row['challenge_sha256'] != challenge_hash
                or row['vector_sha256'] != vector_hash or row['strength'] != -1.
                or any(row.get(k) != v for k, v in expected(plan, tasks[key]).items() if k != 'interventions')
                or row['interventions'] != [dict(layer=row['layer'], group=row['group'], strength=-1.)]):
            raise ValueError('Diagnostic answer differs from frozen question/vector/intervention.')
        done.add(full)
    return rows, done


def steer(plan, output, native):
    import torch
    challenge = json.loads((output/'challenge.json').read_text())
    if challenge['references_sha256'] != digest(output/'references.jsonl'): raise ValueError('Frozen references changed.')
    vector = torch.load(output/'style-vectors.pt', map_location='cpu', weights_only=True)
    config_hash = digest(output/'run_config.json')
    if vector['run_config_sha256'] != config_hash or vector['angry_minus_neutral'].shape != SHAPE:
        raise ValueError('Vector identity/shape differs.')
    if (vector['recipe'] != plan['recipe'] or vector['groups'] != list(GROUPS)
            or vector['collection_question_ids'] != [p['question_id'] for p in plan['collection']]
            or not torch.isfinite(vector['angry_minus_neutral']).all()):
        raise ValueError('Vector collection/model/groups differ or contain invalid values.')
    _, done = completed(plan, output, challenge)
    tasks = {(r['question_id'], r['voice']): r for r in plan['recordings'] if r['style']=='neutral'}
    for layer in plan['layers']:
        for g, group in enumerate(GROUPS):
            for case in challenge['cases']:
                key = case['question_id'], case['voice']
                if (*key, layer, group) in done: continue
                row = tasks[key]
                print(f'Steer L{layer} {group} -1 {row["question_id"]}', flush=True)
                trace = generate(native, row, plan['prompt'], target=layer, group=group,
                    direction=vector['angry_minus_neutral'][layer, g])
                saved = dict(expected(plan, row), **trace, condition=f'neutral-L{layer}-{group}-a-1',
                    layer=layer, group=group, strength=-1., run_config_sha256=config_hash,
                    challenge_sha256=digest(output/'challenge.json'), vector_sha256=digest(output/'style-vectors.pt'))
                saved['interventions'] = [dict(layer=layer, group=group, strength=-1.)]
                with (output/'results.jsonl').open('a') as stream: stream.write(json.dumps(saved)+'\n')


def report(output):
    config = output/'run_config.json'
    if not config.exists(): print('All-layer diagnostic not prepared.'); return
    plan = json.loads(config.read_text())
    print(f'All-layer collection pairs: {len(list((output/"pairs").glob("*.pt")))}/{len(plan["collection"])}')
    _, refs = references(plan, output); print(f'Neutral/angry test references: {len(refs)}/{len(plan["recordings"])}')
    if not (output/'challenge.json').exists(): print('Challenge not selected yet.'); return
    challenge = json.loads((output/'challenge.json').read_text()); rows, _ = completed(plan, output, challenge)
    scored, provenance = score_rows(plan, rows, output)
    print(f'Outcome-selected diagnostic: {len(challenge["cases"])} recoverable test cases; neutral numeric0%, angry100% by selection.')
    print('Layer Group       N  Numeric Flexible Strict Format Mean tokens')
    summaries = []
    for layer in plan['layers']:
        for group in GROUPS:
            items = [r for r in scored if r['layer']==layer and r['group']==group]
            if not items: continue
            n = len(items); metrics = ('flexible-numeric', 'flexible-extract', 'strict-match', 'format-compliant')
            counts = {m: sum(r['scores'][m] for r in items) for m in metrics}
            entry = dict(layer=layer, group=group, n=n, **counts, mean_tokens=statistics.mean(r['response_tokens'] for r in items))
            summaries.append(entry)
            print(f'{layer:5d} {group:7} {n:5d} '+' '.join(f'{counts[m]/n:7.1%}' for m in metrics)+f' {entry["mean_tokens"]:10.1f}')
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    (output/'summary.json').write_text(json.dumps(dict(challenge=challenge, layers=summaries, scorer=provenance,
        note=plan['selection']), indent=2)+'\n')
    if summaries:
        with (output/'layer-summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('prepare', 'collect', 'select', 'steer', 'summary'), default='collect')
    p.add_argument('--collection', type=Path, default=Path('outputs/phi4-gsm-reverse400-400-L30'))
    p.add_argument('--neutral-run', type=Path, default=Path('outputs/phi4-gsm-reverse400-869-L17'))
    p.add_argument('--delivery-run', type=Path, default=Path('outputs/gsm8k-candidates/phi4/original'))
    p.add_argument('--output', type=Path, default=Path('outputs/phi4-gsm400-all-layers-test869-minus1'))
    args = p.parse_args()
    if args.mode == 'summary': report(args.output); return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(args)
        if args.mode == 'prepare': reuse_references(plan, args.output); return
        if args.mode == 'select': select_challenge(plan, args.output); return
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        if args.mode == 'steer':
            challenge = json.loads((args.output/'challenge.json').read_text()); _, done = completed(plan, args.output, challenge)
            if len(done) == len(challenge['cases'])*len(plan['layers'])*len(GROUPS): print('Diagnostic complete; no model loaded.'); return
        candidates.concurrent_budget_now(plan['gpu_budget_gib'])
        native = engine.NativePhi()
        (args.output/f'runtime-{args.mode}.json').write_text(json.dumps(dict(torch=torch.__version__,
            transformers=engine.version('transformers'), gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
            script_sha256=digest(__file__), engine_sha256=digest(engine.__file__), gpu_budget_gib=plan['gpu_budget_gib'],
            note='No global GPU queue; wrapper waits only for current Phi L17 -2.'), indent=2)+'\n')
        if args.mode == 'collect':
            collect_vectors(plan, args.output, native)
            tasks, done = references(plan, args.output)
            for key, row in tasks.items():
                if key in done: continue
                print(f'Reference {row["question_id"]} {row["style"]}', flush=True)
                trace = generate(native, row, plan['prompt'])
                saved = dict(expected(plan, row), **trace, run_config_sha256=digest(args.output/'run_config.json'))
                with (args.output/'references.jsonl').open('a') as stream: stream.write(json.dumps(saved)+'\n')
        else: steer(plan, args.output, native)


if __name__ == '__main__': main()
