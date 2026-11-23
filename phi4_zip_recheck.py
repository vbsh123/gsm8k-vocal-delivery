"""Fresh matched DIFFA/Phi cohorts: collect three directions, evaluate deliveries, then steer with three workers."""
import argparse
import gc
import sys
import fcntl
import json
from pathlib import Path
import subprocess
import time
import statistics

from jenny_bundle_registry import Registry, sha

DATASETS = ('gsm8k', 'multiarith', 'svamp', 'bbh3', 'bbh5', 'ifeval')
PROTOCOL = 'phi4-diffa-exact-cohorts-three-directions-minus1-v4'
TARGETS = {'L0-audio': (0, 'audio'), 'L11-answer': (11, 'answer'), 'L30-answer': (30, 'answer')}
CAPTURE_LAYERS = [0, 11, 30]
CAPTURE_GROUPS = [1, 0, 0]  # Native hook output: answer=0, audio=1.


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
    plan = dict(protocol=PROTOCOL, task=name, name=name, recipe=recipe, targets=TARGETS, capture_layers=CAPTURE_LAYERS, strength=-1.,
                source_config=str(data_path.absolute()), source_config_sha256=sha(data_path),
                collection=[pairs[q] for q in train], test=[pairs[q] for q in test], bank=cohort['bank'],
                prompt=cohort['prompt'], collection_prompt=cohort['collection_prompt'], seed=42,
                max_new_tokens=cohort['max_new_tokens'], collection_max_new_tokens=cohort['collection_max_new_tokens'],
                deliveries=[dict(pairs[q][style], condition=style, split=split)
                            for split, ids in [('collection', train), ('test', test)] for q in ids for style in ('neutral', 'angry')],
                added_delivery_hashes=[], collection_ids=train, test_ids=test,
                cohort_source_plan_sha256=cohort['source_plan_sha256'],
                cohort_match='Exact ordered collection AND evaluation IDs from original DIFFA; exact neutral audio hashes from ZIP.',
                definition='Fresh per-question angry mean minus neutral mean at L0 audio, L11 answer and L30 answer; all native generation forwards including prefill. '
                           'Raw equal-question float32 mean, no e2v/outcome filtering. No prior vector/response reuse for fresh inference.',
                audio_policy='Same ZIP recordings and original DIFFA neutral hashes; no old Phi audio/results used.',
                verification='Fresh neutral and angry evaluated on collection AND held-out IDs before steering; no historical comparisons.',
                note='Fresh ZIP experiment using previously inspected question splits and selected layers; exploratory, not an independent confirmatory benchmark.')
    freeze(folder/'run_config.json', plan)
    print(f'{name}: EXACT DIFFA {len(train)}/{len(test)} split; {2*(len(train)+len(test))} fresh delivery answers, then 3 workers, each {len(test)} steered answers at -1.', flush=True)
    return plan


def saved_rows(path, plan, repair=False):
    from steer import read_rows
    result = {}
    expected = {(r['question_id'], r['condition']): r for r in plan['deliveries']}
    expected.update({(p['question_id'], 'steered'): p['neutral'] for p in plan['test']})
    target = next((k for k in TARGETS if path.name == f'steering-{k}.jsonl'), None)
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
        intervention = [dict(layer=TARGETS[target][0],group=TARGETS[target][1],strength=-1.)] if key[1] == 'steered' and target else []
        if row.get('interventions', []) != intervention or (key[1] == 'steered' and target is None):
            raise ValueError('Saved row intervention does not match its worker file.')
        result[key] = row
    return result


def save_response(path, plan, record, trace, condition, split, target=None):
    row = dict(record, response=trace['response'], response_tokens=trace['response_tokens'],
               generation_seconds=trace.get('generation_seconds'), condition=condition, split=split,
               prompt=plan.get('collection_prompt', plan['prompt']) if split == 'collection' else plan['prompt'],
               seed=42, max_new_tokens=plan.get('collection_max_new_tokens', plan['max_new_tokens']) if split == 'collection' else plan['max_new_tokens'],
               model=plan['recipe']['model'], revision=plan['recipe']['revision'],
               run_config_sha256=sha(path.parent/'run_config.json'), strength=-1. if condition == 'steered' else 0.,
               interventions=[dict(layer=TARGETS[target][0], group=TARGETS[target][1], strength=-1.)] if condition == 'steered' else [])
    with path.open('a') as stream:
        stream.write(json.dumps(row)+'\n')
        stream.flush()
    return row


def report(plan, folder, target=None):
    import winning_diffa_sweeps as winners
    from rescore_audio_steering import paired
    name = plan['task']
    deliveries = list(saved_rows(folder/'deliveries.jsonl', plan).values())
    steering = list(saved_rows(folder/f'steering-{target}.jsonl', plan).values()) if target else []
    scoring_target = dict(task=name, name=name, bank=plan['bank'])
    def score(records):
        scored, provenance = winners.score_rows(dict(target=scoring_target), records, folder)
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
    print(f'\n{name}: fresh ZIP check, {target or "delivery-only"} -1. Delivery {len(deliveries)}/{len(plan["deliveries"])}; steering {len(steering)}/{len(plan["test"])}.', flush=True)
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
    result = dict(task=name, protocol=PROTOCOL, steering_target=target, strength=-1., primary_metric='strict_prompt' if name == 'ifeval' else
                  'flexible-extract' if name.startswith('bbh') else 'flexible-numeric', paired=comparisons,
                  verification=verification, lengths=lengths, scorer=provenance)
    winners.atomic_text(folder/(f'summary-{target}.json' if target else 'delivery-summary.json'), json.dumps(result, indent=2)+'\n')
    winners.atomic_text(folder/(f'rescored-{target}.jsonl' if target else 'delivery-rescored.jsonl'), ''.join(json.dumps(r)+'\n' for r in scored))
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
                                       capture_layers=CAPTURE_LAYERS, max_new_tokens=plan.get('collection_max_new_tokens',plan['max_new_tokens']))
                means = trace.pop('means')
                if means.shape != (3,2,3072) or not torch.isfinite(means).all():
                    raise ValueError('Need all three layer captures from the same generation.')
                trace['mean'] = torch.stack([means[i,g] for i,g in enumerate(CAPTURE_GROUPS)])
                traces[style] = trace
            saved = dict(run_config_sha256=sha(folder/'run_config.json'), question_id=pair['question_id'], traces=traces,
                         difference=traces['angry']['mean']-traces['neutral']['mean'])
            atomic_save(cache, saved)
        if (saved['difference'].shape != (3,3072) or not torch.isfinite(saved['difference']).all()
                or not torch.equal(saved['difference'], saved['traces']['angry']['mean']-saved['traces']['neutral']['mean'])):
            raise ValueError('Invalid fresh three-direction pair difference or sign.')
        differences.append(saved['difference'])
        for style in ('neutral', 'angry'):
            key = pair['question_id'], style
            if key not in done:
                done[key] = save_response(path, plan, pair[style], saved['traces'][style], style, 'collection')
    vector = torch.stack(differences).mean(0)
    vector_path = folder/'style-vectors.pt'
    state = dict(protocol=PROTOCOL, run_config_sha256=sha(folder/'run_config.json'), targets=TARGETS, capture_layers=CAPTURE_LAYERS,
                 model=plan['recipe']['model'], revision=plan['recipe']['revision'],
                 collection_question_ids=[p['question_id'] for p in plan['collection']],
                 directions={key:vector[i] for i,key in enumerate(TARGETS)}, angry_minus_neutral=vector[1])
    if vector_path.exists():
        previous = torch.load(vector_path, map_location='cpu', weights_only=True)
        if previous['run_config_sha256'] != state['run_config_sha256'] or any(not torch.equal(previous['directions'][key], value) for key,value in state['directions'].items()):
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
    score_command(plan, folder, None)
    if phase == 'check':
        return
    steer_task(plan, folder, native, 'L11-answer')


def score_command(plan, folder, target):
    command = [str(Path('.venv-diffa/bin/python').absolute()), '-u', __file__, '--mode', 'summary',
               '--datasets', plan['task'], '--output', str(folder.parent),
               '--summary-target', target or 'deliveries']
    subprocess.run(command, check=True)


def steer_task(plan, folder, native, target):
    import torch
    import phi4_all_layer_diagnostic as hooks
    state = torch.load(folder/'style-vectors.pt',map_location='cpu',weights_only=True)
    if state['protocol'] != PROTOCOL or state['run_config_sha256'] != sha(folder/'run_config.json'):
        raise ValueError('Steering requires the matching completed three-direction collection.')
    expected = {(r['question_id'],r['condition']) for r in plan['deliveries']}
    if set(saved_rows(folder/'deliveries.jsonl',plan)) != expected:
        raise ValueError('Fresh neutral/angry evaluation must finish before steering.')
    vector = state['directions'][target]
    layer, group = TARGETS[target]
    path = folder/f'steering-{target}.jsonl'
    steered = saved_rows(path, plan, repair=True)
    for index, pair in enumerate(plan['test'],1):
        if (pair['question_id'],'steered') in steered:
            continue
        print(f'{plan["task"]} {target} -1 [{index}/{len(plan["test"])}] {pair["question_id"]}',flush=True)
        trace = hooks.generate(native,pair['neutral'],plan['prompt'],target=layer,group=group,
                               direction=vector,strength=-1.,max_new_tokens=plan['max_new_tokens'])
        save_response(path,plan,pair['neutral'],trace,'steered','test',target)
    score_command(plan,folder,target)


def worker(args):
    from evaluate_bbh_candidates import concurrent_budget_now
    from steer_phi4_gsm_reverse import NativePhi
    # Each worker owns one model and runs its direction sequentially over datasets.
    concurrent_budget_now(15.)
    native = NativePhi(TARGETS[args.worker_target][0])
    for name in args.datasets:
        folder = args.output/name
        plan = json.loads((folder/'run_config.json').read_text())
        with (folder/f'{args.worker_target}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            steer_task(plan,folder,native,args.worker_target)


def launch_workers(args):
    processes = []
    for target in TARGETS:
        command = [sys.executable,'-u',__file__,'--worker-target',target,'--output',str(args.output),
                   '--datasets',*args.datasets]
        with (args.output/f'steering-{target}.log').open('a') as log:
            processes.append((target,subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)))
    failures = [(target,code) for target,process in processes if (code := process.wait()) != 0]
    if failures:
        raise RuntimeError(f'Steering workers failed: {failures}; inspect the per-direction logs. Completed answers are resumable.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'run', 'summary'], default='run')
    parser.add_argument('--worker-target',choices=list(TARGETS))
    parser.add_argument('--summary-target',choices=['deliveries',*TARGETS])
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--audio-root', type=Path, default=Path('outputs/jenny-zip-audio'))
    parser.add_argument('--cohorts', type=Path, default=Path('/workspace/diffa-original-cohorts.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/phi4-diffa-cohorts-three-directions'))
    args = parser.parse_args()
    if args.mode == 'summary':
        for name in args.datasets:
            folder = args.output/name
            if (folder/'run_config.json').exists():
                plan = json.loads((folder/'run_config.json').read_text())
                for target in ([None] if args.summary_target == 'deliveries' else [args.summary_target] if args.summary_target else list(TARGETS)):
                    report(plan, folder, target)
            else:
                print(name, 'not prepared.')
        return
    if args.worker_target is not None:
        worker(args)
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
            print('All datasets completed three-direction collection and full delivery evaluation. Releasing collection model, then starting three -1 steering workers.',flush=True)
            del native
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            launch_workers(args)


if __name__ == '__main__':
    main()
