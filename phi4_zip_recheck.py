"""Fresh angry L11/-1 Phi collection and full delivery check from the Jenny ZIP, then held-out steering."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
import statistics

from jenny_bundle_registry import Registry, sha

DATASETS = ('gsm8k', 'multiarith', 'svamp', 'bbh3', 'bbh5', 'ifeval')
PROTOCOL = 'phi4-diffa-exact-cohorts-L11-answer-minus1-v3'


def prepare(name, data_path, registry, folder):
    import evaluate_gsm_candidates as gsm
    from winning_diffa_sweeps import freeze
    from export_diffa_cohorts import PROTOCOL as COHORT_PROTOCOL
    manifest = json.loads(data_path.read_text())
    if manifest.get('protocol') != COHORT_PROTOCOL:
        raise ValueError('Need the exported original DIFFA cohort file, not old Phi splits.')
    cohort = manifest['datasets'][name]
    train, test = cohort['collection_ids'], cohort['test_ids']
    if (cohort['task'] != name or cohort['seed'] != 42 or not train or not test
            or len(set(train+test)) != len(train+test)):
        raise ValueError('Invalid or overlapping original DIFFA cohorts.')
    pairs = {}
    for q in train+test:
        example = cohort['bank'][q]
        question = example.get('question', example.get('prompt'))
        answer = None if name == 'ifeval' else str(example['answer']).rsplit('####', 1)[-1].strip()
        entries = {}
        for style in ('neutral', 'angry'):
            expected = cohort['neutral'][q]['audio_sha256'] if style == 'neutral' else None
            item = registry.choose(name, q, style, question=question, answer=answer, expected=expected)
            if item is None:
                raise ValueError(f'ZIP lacks exact DIFFA-compatible {name}/{q}/{style}; no new cohort or old-Phi audio substituted.')
            options = registry.options.get((name,q,style), [])
            compatible = {x['record']['audio_sha256'] for x in options
                          if x['question'] == question and (answer is None or x['answer'] == answer)}
            if style == 'angry' and len(compatible) > 1:
                raise ValueError(f'Multiple angry recordings for {name}/{q}; pin the DIFFA recording before running.')
            entries[style] = item['record']
        pairs[q] = dict(question_id=q, voice=entries['neutral']['voice'], **entries)
    recipe = dict(gsm.MODELS['phi4'], dtype='bfloat16', do_sample=False, num_beams=1,
                  use_cache=True, repetition_penalty=1., text_only_output=True)
    plan = dict(protocol=PROTOCOL, task=name, name=name, recipe=recipe, layer=11, group='answer', strength=-1.,
                source_config=str(data_path.absolute()), source_config_sha256=sha(data_path),
                collection=[pairs[q] for q in train], test=[pairs[q] for q in test], bank=cohort['bank'],
                prompt=cohort['prompt'], collection_prompt=cohort['collection_prompt'], seed=42,
                max_new_tokens=cohort['max_new_tokens'], collection_max_new_tokens=cohort['collection_max_new_tokens'],
                deliveries=[dict(pairs[q][style], condition=style, split=split)
                            for split, ids in [('collection', train), ('test', test)] for q in ids for style in ('neutral', 'angry')],
                added_delivery_hashes=[], collection_ids=train, test_ids=test,
                cohort_source_plan_sha256=cohort['source_plan_sha256'],
                cohort_match='Exact ordered collection AND evaluation IDs from original DIFFA; exact neutral audio hashes from ZIP.',
                definition='Fresh per-question angry mean minus neutral mean at L11 answer; all native generation forwards including prefill. '
                           'Raw equal-question float32 mean, no e2v/outcome filtering. No prior vector/response reuse for fresh inference.',
                audio_policy='Same ZIP recordings and original DIFFA neutral hashes; no old Phi audio/results used.',
                verification='Fresh neutral and angry evaluated on collection AND held-out IDs before steering; no historical comparisons.',
                note='Fresh ZIP experiment using previously inspected question splits and selected L11; exploratory, not an independent confirmatory benchmark.')
    freeze(folder/'run_config.json', plan)
    print(f'{name}: EXACT DIFFA {len(train)}/{len(test)} split; {2*(len(train)+len(test))} fresh delivery answers, then {len(test)} L11/-1 steered answers.', flush=True)
    return plan


def saved_rows(path, plan, repair=False):
    from steer import read_rows
    result = {}
    expected = {(r['question_id'], r['condition']): r for r in plan['deliveries']}
    expected.update({(p['question_id'], 'steered'): p['neutral'] for p in plan['test']})
    for row in read_rows(path, repair=repair):
        key = row['question_id'], row['condition']
        if key not in expected or key in result or row.get('run_config_sha256') != sha(path.parent/'run_config.json'):
            raise ValueError(f'Duplicate or incompatible row in {path}')
        collection = expected[key].get('split') == 'collection'
        prompt = plan.get('collection_prompt', plan['prompt']) if collection else plan['prompt']
        budget = plan.get('collection_max_new_tokens', plan['max_new_tokens']) if collection else plan['max_new_tokens']
        if row['prompt'] != prompt or row['model'] != plan['recipe']['model'] or row['revision'] != plan['recipe']['revision']:
            raise ValueError('Saved fresh-check model or prompt changed.')
        if (row['audio_sha256'] != expected[key]['audio_sha256'] or row['seed'] != 42
                or row['max_new_tokens'] != budget
                or row['strength'] != (-1. if key[1] == 'steered' else 0.)):
            raise ValueError('Saved fresh-check audio or generation recipe changed.')
        result[key] = row
    return result


def save_response(path, plan, record, trace, condition, split):
    row = dict(record, response=trace['response'], response_tokens=trace['response_tokens'],
               generation_seconds=trace.get('generation_seconds'), condition=condition, split=split,
               prompt=plan.get('collection_prompt', plan['prompt']) if split == 'collection' else plan['prompt'],
               seed=42, max_new_tokens=plan.get('collection_max_new_tokens', plan['max_new_tokens']) if split == 'collection' else plan['max_new_tokens'],
               model=plan['recipe']['model'], revision=plan['recipe']['revision'],
               run_config_sha256=sha(path.parent/'run_config.json'), strength=-1. if condition == 'steered' else 0.,
               interventions=[dict(layer=11, group='answer', strength=-1.)] if condition == 'steered' else [])
    with path.open('a') as stream:
        stream.write(json.dumps(row)+'\n')
        stream.flush()
    return row


def report(plan, folder):
    import winning_diffa_sweeps as winners
    from rescore_audio_steering import paired
    name = plan['task']
    deliveries = list(saved_rows(folder/'deliveries.jsonl', plan).values())
    steering = list(saved_rows(folder/'steering.jsonl', plan).values())
    target = dict(task=name, name=name, bank=plan['bank'])
    def score(records):
        scored, provenance = winners.score_rows(dict(target=target), records, folder)
        if name.startswith('bbh') and records:
            from score_bbh_harness import HarnessScorer
            from evaluate_bbh_audio_models import strict_format_bits
            scorer = HarnessScorer(folder/'harness/upstream', objects='three' if name == 'bbh3' else 'five')
            docs = [dict(input=plan['bank'][r['question_id']]['original_input'], target=plan['bank'][r['question_id']]['original_target']) for r in records]
            bits = strict_format_bits(scorer, [r['response'] for r in records], docs, 'ABC' if name == 'bbh3' else 'ABCDE')
            for row, bit in zip(scored, bits):
                row['metrics']['format-compliant'] = bit
        return scored, provenance
    scored, provenance = score(deliveries+steering)
    groups = {c: {r['question_id']: r for r in scored if r['condition'] == c} for c in ('neutral', 'angry', 'steered')}
    comparisons = []
    print(f'\n{name}: fresh ZIP check, L11 answer -1. Delivery {len(deliveries)}/{len(plan["deliveries"])}; steering {len(steering)}/{len(plan["test"])}.', flush=True)
    for split, ids in [('collection', [p['question_id'] for p in plan['collection']]), ('held-out', [p['question_id'] for p in plan['test']]),
                       ('all', [p['question_id'] for p in plan['collection']+plan['test']])]:
        for condition in (('angry', 'steered') if split == 'held-out' else ('angry',)):
            common = sorted(set(ids) & set(groups['neutral']) & set(groups[condition]))
            if not common:
                continue
            for metric in groups['neutral'][common[0]]['metrics']:
                value = paired([(groups['neutral'][q]['metrics'][metric], groups[condition][q]['metrics'][metric]) for q in common])
                comparisons.append(dict(split=split, condition=condition, metric=metric, **value))
                print(f'{split:10} {condition:8} {metric:18} N={len(common)}; '
                      f'{value["baseline_correct"]}/{len(common)} -> {value["steered_correct"]}/{len(common)}; '
                      f'gain={value["gain_pp"]:+.2f}; p={value["mcnemar_exact_p"]:.4g}; CI={value["ci95_gain_pp"]}', flush=True)
    verification = dict(matched=0, expected=len(plan['deliveries']), differences=[],
                        historical_comparison=False, added_delivery_hashes=plan['added_delivery_hashes'])
    print('Fresh ZIP audio only: no historical responses or vectors loaded.', flush=True)
    lengths = {}
    for condition, values in groups.items():
        counts = [r['response_tokens'] for r in values.values()]
        if counts:
            lengths[condition] = dict(n=len(counts), mean=statistics.mean(counts), median=statistics.median(counts))
            print(condition, 'mean/median tokens:', round(lengths[condition]['mean'],1), lengths[condition]['median'], flush=True)
    result = dict(task=name, protocol=PROTOCOL, primary_metric='strict_prompt' if name == 'ifeval' else
                  'flexible-extract' if name.startswith('bbh') else 'flexible-numeric', paired=comparisons,
                  verification=verification, lengths=lengths, scorer=provenance)
    winners.atomic_text(folder/'summary.json', json.dumps(result, indent=2)+'\n')
    winners.atomic_text(folder/'rescored.jsonl', ''.join(json.dumps(r)+'\n' for r in scored))
    return verification


def run_task(plan, folder, native, phase='all'):
    import torch
    import phi4_all_layer_diagnostic as hooks
    from kimi_bbh_layer_diagnostic import atomic_save
    (folder/'pairs').mkdir(exist_ok=True)
    path = folder/'deliveries.jsonl'
    done = saved_rows(path, plan, repair=True)
    differences = []
    for index, pair in enumerate(plan['collection']):
        cache = folder/'pairs'/f'{index:04d}.pt'
        if cache.exists():
            saved = torch.load(cache, map_location='cpu', weights_only=True)
            if saved['run_config_sha256'] != sha(folder/'run_config.json') or saved['question_id'] != pair['question_id']:
                raise ValueError('Fresh collection cache changed.')
        else:
            print(f'{plan["task"]} fresh collect [{index+1}/{len(plan["collection"])}] {pair["question_id"]}', flush=True)
            traces = {}
            for style in ('neutral', 'angry'):
                trace = hooks.generate(native, pair[style], plan.get('collection_prompt',plan['prompt']), collect=True,
                                       capture_layers=[11], max_new_tokens=plan.get('collection_max_new_tokens',plan['max_new_tokens']))
                trace['mean'] = trace.pop('means')[0,0]
                traces[style] = trace
            saved = dict(run_config_sha256=sha(folder/'run_config.json'), question_id=pair['question_id'], traces=traces,
                         difference=traces['angry']['mean']-traces['neutral']['mean'])
            atomic_save(cache, saved)
        if (saved['difference'].shape != (3072,) or not torch.isfinite(saved['difference']).all()
                or not torch.equal(saved['difference'], saved['traces']['angry']['mean']-saved['traces']['neutral']['mean'])):
            raise ValueError('Invalid fresh L11 pair difference or sign.')
        differences.append(saved['difference'])
        for style in ('neutral', 'angry'):
            key = pair['question_id'], style
            if key not in done:
                done[key] = save_response(path, plan, pair[style], saved['traces'][style], style, 'collection')
    vector = torch.stack(differences).mean(0)
    vector_path = folder/'style-vectors.pt'
    state = dict(protocol=PROTOCOL, run_config_sha256=sha(folder/'run_config.json'), layer=11, group='answer',
                 model=plan['recipe']['model'], revision=plan['recipe']['revision'],
                 collection_question_ids=[p['question_id'] for p in plan['collection']], angry_minus_neutral=vector)
    if vector_path.exists():
        previous = torch.load(vector_path, map_location='cpu', weights_only=True)
        if previous['run_config_sha256'] != state['run_config_sha256'] or not torch.equal(previous['angry_minus_neutral'], vector):
            raise ValueError('Fresh recomputed vector differs from saved vector.')
    else:
        atomic_save(vector_path, state)
    for record in plan['deliveries']:
        key = record['question_id'], record['condition']
        if key in done:
            continue
        print(f'{plan["task"]} fresh delivery [{len(done)+1}/{len(plan["deliveries"])}] {key}', flush=True)
        trace = hooks.generate(native, record, plan['prompt'], max_new_tokens=plan['max_new_tokens'])
        done[key] = save_response(path, plan, record, trace, record['condition'], record['split'])
    # Score the full fresh delivery run BEFORE performing any steering.
    command = [str(Path('.venv-diffa/bin/python').absolute()), '-u', __file__, '--mode', 'summary',
               '--datasets', plan['task'], '--output', str(folder.parent)]
    subprocess.run(command, check=True)
    verification = json.loads((folder/'summary.json').read_text())['verification']
    if phase == 'check':
        return verification
    steered = saved_rows(folder/'steering.jsonl', plan, repair=True)
    for index, pair in enumerate(plan['test'], 1):
        q = pair['question_id']
        if (q, 'steered') in steered:
            continue
        print(f'{plan["task"]} L11 answer -1 [{index}/{len(plan["test"])}] {q}', flush=True)
        trace = hooks.generate(native, pair['neutral'], plan['prompt'], target=11, group='answer',
                               direction=vector, strength=-1., max_new_tokens=plan['max_new_tokens'])
        save_response(folder/'steering.jsonl', plan, pair['neutral'], trace, 'steered', 'test')
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'run', 'summary'], default='run')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--audio-root', type=Path, default=Path('outputs/jenny-zip-audio'))
    parser.add_argument('--cohorts', type=Path, default=Path('/workspace/diffa-original-cohorts.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/phi4-diffa-cohorts-L11-minus1'))
    args = parser.parse_args()
    if args.mode == 'summary':
        for name in args.datasets:
            folder = args.output/name
            if (folder/'run_config.json').exists():
                report(json.loads((folder/'run_config.json').read_text()), folder)
            else:
                print(name, 'not prepared.')
        return
    if not args.cohorts.is_file():
        raise FileNotFoundError('Copy /workspace/diffa-original-cohorts.json from original DIFFA first; run export_diffa_cohorts.py there.')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'queue.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = None
        plans = []
        for name in args.datasets:
            folder = args.output/name
            folder.mkdir(exist_ok=True)
            config = folder/'run_config.json'
            if config.exists():
                plan = json.loads(config.read_text())
                if plan['protocol'] != PROTOCOL or sha(Path(plan['source_config'])) != plan['source_config_sha256']:
                    raise ValueError('Fresh Phi source protocol/split changed.')
                for record in plan['deliveries']:
                    if sha(Path(record['audio_file'])) != record['audio_sha256']:
                        raise ValueError(f'Frozen ZIP audio changed: {record["question_id"]}/{record["condition"]}')
            else:
                registry = registry or Registry(args.audio_root)
                plan = prepare(name, args.cohorts, registry, folder)
            plans.append((plan, folder))
        if args.mode == 'prepare':
            return
        with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
            print('Waiting for shared Phi GPU queue and idle GPU.', flush=True)
            fcntl.flock(gpu, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                time.sleep(30)
            from steer_phi4_gsm_reverse import NativePhi
            native = NativePhi(11)
            for plan, folder in plans:
                run_task(plan, folder, native, phase='check')
            print('All selected datasets have completed fresh collection and full delivery checks. Starting held-out L11/-1 steering.', flush=True)
            for plan, folder in plans:
                run_task(plan, folder, native)


if __name__ == '__main__':
    main()
