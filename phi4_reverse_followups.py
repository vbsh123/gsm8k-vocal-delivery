"""Remaining GSM8K and own-vector MultiArith follow-ups at chosen Phi layers."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import steer_phi4_gsm_reverse as engine
from kimi_bbh_layer_diagnostic import atomic_save, digest, freeze
import run_multiarith_mean as multi
from steer import read_rows


def folders(layer, dataset):
    return Path('outputs')/(f'phi4-gsm-reverse400-869-L{layer}' if dataset == 'gsm' else f'phi4-multiarith-reverse290-290-L{layer}')


def verify_receipts(plan):
    for r in plan.get('source_receipts', []):
        if digest(r['path']) != r['sha256']: raise ValueError('Frozen source evidence changed: '+r['path'])


def receipts(paths):
    return [dict(path=str(p.absolute()), sha256=digest(p)) for p in paths]


def prepare_gsm(source, previous, output, layer):
    config = output/'run_config.json'
    if config.exists():
        plan = json.loads(config.read_text()); verify_receipts(plan)
        if plan['layer'] != layer or plan['fixed_vector_source'] != str((previous/'style-vectors.pt').absolute()):
            raise ValueError('Different GSM continuation settings.')
        return plan
    old_config = previous/'run_config.json'; old = json.loads(old_config.read_text())
    if (old['protocol'] != engine.protocol_for(layer) or old['layer'] != layer or old['prompt'] != engine.gsm.ORIGINAL
            or len(old['collection']) != 400 or len(old['test']) != 400 or old['strengths'] != engine.strengths(layer)):
        raise ValueError('Need the completed matching400/400 Phi experiment.')
    _, done = engine.completed(old, previous, digest(old_config))
    if len(done) != 1200: raise ValueError('Finish all1200 original400-question answers before extension.')
    parent = engine.gsm.make_plan(source, 'phi4', 'original')
    if parent['recipe'] != old['recipe']: raise ValueError('Phi recipe differs from original run.')
    used = {p['question_id'] for p in old['collection']}|{r['question_id'] for r in old['test']}
    questions = [q for q in engine.gsm.questions() if q['question_id'] not in used]
    if len(used) != 800 or len(questions) != 469: raise ValueError('Expected400 collection,400 original test and469 remaining.')
    records = {(r['question_id'], r['style']): r for r in parent['recordings']}
    test = old['test']+[records[q['question_id'], 'neutral'] for q in questions]
    bank = {q['question_id']: q for q in engine.gsm.questions() if q['question_id'] in {r['question_id'] for r in test}}
    plan = dict(old, protocol=f'phi4-gsm8k-frozen400-test869-L{layer}-reverse-v1', dataset='GSM8K',
                source=str(source.absolute()), test=test, bank=bank, original_questions=400, new_questions=469,
                fixed_vector_source=str((previous/'style-vectors.pt').absolute()), previous_run=str(previous.absolute()),
                source_receipts=receipts([old_config, previous/'results.jsonl', previous/'style-vectors.pt']),
                selection=old['selection']+' Continuation retains original400 test answers and adds all469 remaining '
                    'non-collection questions, with frozen vector/layer/strengths; no new extraction or outcome filtering.')
    freeze(config, plan)
    print(f'GSM L{layer}: retain400 tests; add469 new tests;869 total. Frozen400-question vector reused.', flush=True)
    return plan


def prepare_gsm_extra(previous, output, layer, strength):
    if not math.isfinite(strength) or strength >= 0:
        raise ValueError('Extra reverse strength must be finite and negative.')
    old_config = previous/'run_config.json'
    old = json.loads(old_config.read_text())
    verify_receipts(old)
    if old['layer'] != layer or len(old['test']) != 869 or old['prompt'] != engine.gsm.ORIGINAL:
        raise ValueError('Need matching completed869-question GSM continuation.')
    _, done = engine.completed(old, previous, digest(old_config))
    if len(done) != len(old['test'])*len(old['strengths']):
        raise ValueError('Finish original869-question continuation before extra strength.')
    plan = dict(old, protocol=f'phi4-gsm8k-frozen400-test869-L{layer}-extra-{strength:g}-v1',
                strengths={'neutral': 0., f'neutral-L{layer}-answer-a{strength:g}': strength},
                source_receipts=old.get('source_receipts', [])+receipts([old_config, previous/'results.jsonl', previous/'style-vectors.pt']),
                selection=old['selection']+' Exploratory additional strength selected after inspecting earlier strengths; same869 tests and frozen vector.')
    freeze(output/'run_config.json', plan)
    return plan


def install_fixed_vector(plan, output):
    import torch
    verify_receipts(plan)
    source = Path(plan['fixed_vector_source'])
    v = torch.load(source, map_location='cpu', weights_only=True)
    old = json.loads((Path(plan['previous_run'])/'run_config.json').read_text())
    if (v['run_config_sha256'] != digest(Path(plan['previous_run'])/'run_config.json') or v['layer'] != plan['layer']
            or v['collection_question_ids'] != [p['question_id'] for p in plan['collection']]
            or v['revision'] != plan['recipe']['revision'] or v['protocol'] != old['protocol']
            or v['angry_minus_neutral'].shape != (3072,) or not torch.isfinite(v['angry_minus_neutral']).all()):
        raise ValueError('Frozen vector identity/layer differs.')
    path = output/'style-vectors.pt'
    if path.exists():
        if digest(path) != digest(source): raise ValueError('Continuation vector differs from frozen original.')
    else:
        temp = path.with_suffix('.tmp'); shutil.copyfile(source, temp); temp.replace(path)
    return v['angry_minus_neutral']


def multi_selection(source):
    path = source/'MultiArith-source.json'
    if not path.exists():
        data = multi.reasoning.download(multi.SOURCE_URL)
        multi.split_questions(data); source.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    train, test = multi.split_questions(path.read_bytes())
    bank = {q['question_id']: q for q in train+test}
    raw = multi.reasoning.source_records(source, bank, ['neutral', 'angry'])
    records = {k: dict(r, reference_answer=r['answer']) for k, r in raw.items()}
    for r in records.values(): r.pop('answer')
    pairs = []
    for q in train:
        n, a = records[q['question_id'], 'neutral'], records[q['question_id'], 'angry']
        if n['voice'] != a['voice']: raise ValueError('MultiArith collection voices differ.')
        pairs.append(dict(question_id=q['question_id'], voice=n['voice'], neutral=n, angry=a))
    return pairs, [records[q['question_id'], 'neutral'] for q in test], bank


def prepare_multi(source, output, layer, shared):
    config = output/'run_config.json'
    if config.exists():
        plan = json.loads(config.read_text())
        if plan['layer'] != layer or plan['shared_collection'] != str(shared.absolute()):
            raise ValueError('Different MultiArith settings.')
        return plan
    pairs, test, bank = multi_selection(source)
    if {p['question_id'] for p in pairs}&{r['question_id'] for r in test}: raise ValueError('MultiArith split overlap.')
    plan = dict(protocol=f'phi4-multiarith-own290-test290-L{layer}-reverse-v1', dataset='MultiArith',
                source=str(source.absolute()), source_sha256=multi.SOURCE_SHA256, recipe=dict(engine.MODELS['phi4'],
                    dtype='bfloat16', do_sample=False, num_beams=1, use_cache=True, repetition_penalty=1., text_only_output=True),
                prompt=engine.gsm.ORIGINAL, seed=42, max_new_tokens=512, layer=layer, group='answer',
                strengths=engine.strengths(layer), collection=pairs, test=test,
                bank={r['question_id']: bank[r['question_id']] for r in test},
                collection_bank={p['question_id']: bank[p['question_id']] for p in pairs}, shared_collection=str(shared.absolute()),
                vector_definition='Own MultiArith angry minus neutral: equal per-question mean of native answer-prediction '
                    'generation-forward means on290 collection pairs. Raw magnitude; no normalization. '
                    'L17 and L30 observed in SAME native generation and cached once; selected layer controls evaluation.',
                selection='Pinned600 original rows,580 unique questions; seed42 split290/290; no outcome/classifier/duration filtering.',
                scoring='Installed upstream GSM8K filters/metric on MultiArith references; numeric equivalence additional. '
                    'Audio-adapted inference, not an official MultiArith harness task. Tests/intervals unadjusted.')
    freeze(config, plan)
    return plan


def collect_multi(plan, output, native, config_hash):
    import torch
    shared = Path(plan['shared_collection']); shared.mkdir(parents=True, exist_ok=True)
    common = dict(protocol='phi4-multiarith-dual-layer-native-mean-v1', collection=plan['collection'],
                  prompt=plan['prompt'], recipe=plan['recipe'], seed=42, max_new_tokens=512, observed_layers=[17, 30])
    common_path = shared/'run_config.json'; freeze(common_path, common); common_hash = digest(common_path)
    folder = shared/'pairs'; folder.mkdir(exist_ok=True)
    means = []
    for i, pair in enumerate(plan['collection']):
        path = folder/f'{i:04d}.pt'
        if path.exists(): saved = torch.load(path, map_location='cpu', weights_only=True)
        else:
            print(f'MultiArith collect [{i+1}/290] {pair["question_id"]}: neutral/angry, L17 and L30 together.', flush=True)
            traces = {s: native.generate(pair[s], plan['prompt'], observe=True, observe_layers=[17, 30]) for s in ('neutral', 'angry')}
            saved = dict(protocol=common['protocol'], run_config_sha256=common_hash, question_id=pair['question_id'],
                         voice=pair['voice'], audio_sha256={s: pair[s]['audio_sha256'] for s in traces}, traces=traces)
            atomic_save(path, saved)
        if (saved['run_config_sha256'] != common_hash or saved['protocol'] != common['protocol']
                or saved['question_id'] != pair['question_id'] or saved['voice'] != pair['voice']
                or saved['audio_sha256'] != {s: pair[s]['audio_sha256'] for s in ('neutral', 'angry')}):
            raise ValueError('Shared collection cache differs from fixed MultiArith split.')
        for s in ('neutral', 'angry'):
            if set(saved['traces'][s]['means']) != {17, 30} or saved['traces'][s]['hook_forwards'] <= 0:
                raise ValueError('Invalid multi-layer collection metadata.')
            for h in saved['traces'][s]['means'].values():
                if h.shape != (3072,) or not torch.isfinite(h).all(): raise ValueError('Invalid MultiArith activation mean.')
        means.append(saved['traces']['angry']['means'][plan['layer']]-saved['traces']['neutral']['means'][plan['layer']])
    vector = dict(protocol=plan['protocol'], run_config_sha256=config_hash, layer=plan['layer'], group='answer',
                  collection_question_ids=[p['question_id'] for p in plan['collection']], angry_minus_neutral=torch.stack(means).mean(0),
                  definition=plan['vector_definition'], revision=plan['recipe']['revision'], model=plan['recipe']['model'])
    target = output/'style-vectors.pt'
    if target.exists():
        previous = torch.load(target, map_location='cpu', weights_only=True)
        if previous['run_config_sha256'] != config_hash or not torch.equal(previous['angry_minus_neutral'], vector['angry_minus_neutral']):
            raise ValueError('Saved MultiArith vector changed.')
    else: atomic_save(target, vector)
    return vector['angry_minus_neutral']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('gsm-remaining', 'gsm-extra', 'multiarith', 'summary-gsm', 'summary-multiarith'), required=True)
    p.add_argument('--strength', type=float, default=-1.5)
    p.add_argument('--concurrent', action='store_true', help='Extra GSM only: allow MiniCPM alongside Phi, with20 GiB Torch cap.')
    p.add_argument('--layer', type=int, choices=(17, 30), required=True)
    p.add_argument('--gsm-source', type=Path, default=Path('outputs/azure-kimi-gsm8k1269'))
    p.add_argument('--multi-source', type=Path, default=Path('outputs/azure-multiarith580'))
    p.add_argument('--gsm-previous', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--shared-collection', type=Path, default=Path('outputs/phi4-multiarith-reverse290-shared'))
    args = p.parse_args(); dataset = 'gsm' if 'gsm' in args.stage else 'multiarith'
    if args.concurrent and args.stage != 'gsm-extra': p.error('--concurrent is only supported for gsm-extra.')
    output = args.output or folders(args.layer, dataset)
    if args.stage == 'gsm-extra' and args.output is None:
        output = Path(str(output)+f'-a{args.strength:g}')
    if args.stage.startswith('summary'):
        engine.report(output); return
    output.mkdir(parents=True, exist_ok=True)
    with (output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if dataset == 'gsm':
            previous = args.gsm_previous or (folders(args.layer, 'gsm') if args.stage == 'gsm-extra'
                        else Path('outputs')/f'phi4-gsm-reverse400-400-L{args.layer}')
            plan = (prepare_gsm_extra(previous, output, args.layer, args.strength) if args.stage == 'gsm-extra'
                    else prepare_gsm(args.gsm_source, previous, output, args.layer))
            vector = install_fixed_vector(plan, output)
            config_hash = digest(output/'run_config.json')
            engine.reuse_answers(plan, output, config_hash, previous)
            other = folders(30 if args.layer == 17 else 17, 'gsm')
        else:
            plan = prepare_multi(args.multi_source, output, args.layer, args.shared_collection)
            config_hash = digest(output/'run_config.json')
            other = folders(30 if args.layer == 17 else 17, 'multiarith')
        engine.reuse_answers(plan, output, config_hash, other, baseline_only=True)
        _, done = engine.completed(plan, output, config_hash, repair=True)
        if len(done) == len(plan['strengths'])*len(plan['test']): print('All test answers retained; no GPU/model loading.'); return
        with Path('outputs/jenny-task-vectors.lock').open('w') as gpu:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(gpu, fcntl.LOCK_EX)
            if args.concurrent:
                import torch
                free, total = torch.cuda.mem_get_info(0)
                if free < 18*2**30: raise RuntimeError('Concurrent Phi needs18 GiB currently free; no jobs stopped.')
                torch.cuda.set_per_process_memory_fraction(20*2**30/total, device=0)
                print('Concurrent Phi:20 GiB Torch cap; allowing MiniCPM GPU work.', flush=True)
            else:
                while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                    print('Waiting for active GPU work.', flush=True); time.sleep(30)
            native = engine.NativePhi(args.layer)
            import torch
            (output/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__, transformers=engine.version('transformers'),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), collector_sha256=digest(__file__), engine_sha256=digest(engine.__file__),
                generation='Unchanged native Phi greedy /BF16 SDPA /cache /seed42 /512 tokens.'), indent=2)+'\n')
            if dataset == 'multiarith': vector = collect_multi(plan, output, native, config_hash)
            engine.evaluate(plan, output, native, vector, config_hash)


if __name__ == '__main__': main()
