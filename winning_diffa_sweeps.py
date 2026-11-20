"""Existing DIFFA winners: own-vector signed strengths and off-diagonal transfer.

No TTS, recollection, cohort selection, vector normalization or prompt changes.
Both queues use the existing L31 answer hook, target recipe and target evaluators.
"""

import argparse
import csv
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import subprocess
import time
import wave

from steer import install_interventions, read_rows

STRENGTHS = tuple(i/2 for i in range(-4, 5))
PROTOCOL = 'diffa-winners-L31-answer-fixed-mean-v1'
OWN_CONDITIONS = {'bbh3': 'neutral-BBH-L31-answer-a1', 'ifeval': 'neutral-IFEval-L31-answer-a1'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Frozen settings changed: {path}. Use a new output directory.')
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(path, json.dumps(value, indent=2)+'\n')


def atomic_text(path, text):
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def config_file(folder):
    for name in ('run_config.json', 'split.json'):
        path = folder/name
        if path.exists():
            return path
    raise FileNotFoundError(f'No frozen run_config.json or split.json: {folder}')


def neutral_records(config):
    if 'test' in config:
        return [p['neutral'] for p in config['test']]
    return [r for r in config['recordings'] if r.get('condition', r.get('style')) == 'neutral']


def vector_path(folder, config, override=None):
    return Path(override or config.get('vector') or folder/'style-vectors.pt').absolute()


def read_vector(path, emotion='angry'):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('mode', 'fixed') != 'fixed' or state.get('method', 'mean') != 'mean':
        raise ValueError(f'Expected the original fixed DIFFA mean vector: {path}')
    try:
        count = int(state['pair_counts'][emotion][31])
        vector = state['pair_mean_sums'][emotion][31, state['groups'].index('answer')].float()/count
    except (KeyError, IndexError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f'Missing DIFFA L31 answer mean: {path}') from error
    if count <= 0 or vector.ndim != 1 or not torch.isfinite(vector).all():
        raise ValueError(f'Invalid L31 answer vector: {path}')
    return vector, dict(extraction_ids=list(state.get('extraction_question_ids', [])),
        collection_prompt=state.get('prompt'), collection_tokens=state.get('max_new_tokens'),
        pairs=count, width=vector.numel())


def load_dataset(name, entry):
    folder = Path(entry['run']).absolute()
    path = config_file(folder)
    config = json.loads(path.read_text())
    if (config['seed'] != 42 or config['decoder'] != 'fast_dllm'
            or config.get('steps', 128) != 128 or config.get('block_length', 32) != 32
            or config.get('temperature', 0.) != 0.):
        raise ValueError(f'Unexpected frozen generation recipe: {name}')
    if config.get('layer', 31) != 31 or config.get('group', 'answer') != 'answer':
        raise ValueError(f'Expected L31 answer experiment: {name}')
    task = entry['task']
    raw = neutral_records(config)
    ids = {r['question_id'] for r in raw}
    if 'bank' in config:
        if set(config['bank']) != ids:
            raise ValueError(f'Neutral recordings do not cover the full frozen target bank: {name}')
        bank = {q: config['bank'][q] for q in ids}
        bank_path = None
    else:
        bank_path = Path(entry['source'])/'questions.jsonl'
        bank = {r['question_id']: r for r in read_rows(bank_path) if r['question_id'] in ids}
    if set(bank) != ids or not raw:
        raise ValueError(f'Target question bank/cohort incomplete: {name}')
    required = ({'key', 'prompt', 'instruction_id_list', 'kwargs'} if task == 'ifeval' else
                {'answer', 'original_input', 'original_target'} if task in ('bbh3', 'bbh5') else {'answer'})
    if any(not required <= q.keys() for q in bank.values()):
        raise ValueError(f'Target references lack fields required by the original evaluator: {name}')
    records, keys = [], set()
    for row in raw:
        audio = Path(row.get('audio_file') or row['audio_path'])
        if not audio.is_file() and entry.get('source'):
            audio = Path(entry['source'])/row['audio_path']
        audio = audio.absolute()
        if digest(audio) != row['audio_sha256']:
            raise ValueError(f'Audio missing/changed: {name}, {audio}')
        with wave.open(str(audio)) as stream:
            seconds = stream.getnframes()/stream.getframerate()
        if seconds <= 0 or (task == 'ifeval' and seconds > 30):
            raise ValueError(f'Audio duration differs from target eligibility: {name}, {audio}')
        key = row['question_id'], row['voice']
        if key in keys:
            raise ValueError(f'Duplicate neutral target recording: {name}, {key}')
        keys.add(key)
        answer = '' if task == 'ifeval' else bank[key[0]].get('answer', row.get('answer', ''))
        records.append(dict(question_id=key[0], voice=key[1], audio_file=str(audio),
            audio_sha256=row['audio_sha256'], seconds=seconds,
            reference_answer=str(answer).rsplit('####', 1)[-1].strip()))
    vector = vector_path(folder, config, entry.get('vector'))
    _, info = read_vector(vector)
    extraction_ids = [p['question_id'] for p in config.get('extraction', [])]
    if extraction_ids and info['extraction_ids'] and set(extraction_ids) != set(info['extraction_ids']):
        raise ValueError(f'Vector collection IDs differ from its original split: {name}')
    training = sorted(set(info['extraction_ids'] or extraction_ids))
    if set(training) & ids:
        raise ValueError(f'Own vector collection overlaps target evaluation questions: {name}')
    if info['collection_prompt'] != config.get('collection_prompt', config['prompt']):
        raise ValueError(f'Vector prompt differs from its original collection recipe: {name}')
    if info['collection_tokens'] != config['max_new_tokens']:
        raise ValueError(f'Vector collection budget differs from its original run: {name}')
    revision = config.get('revision')
    if task == 'ifeval':
        from run_spoken_ifeval import REVISION
        if revision != REVISION:
            raise ValueError('IFEval scorer revision differs from the original experiment.')
    primary = 'strict_prompt' if task == 'ifeval' else ('flexible-extract' if task.startswith('bbh') else 'flexible-numeric')
    return dict(name=name, task=task, run=str(folder), config_file=str(path), config_sha256=digest(path),
        bank_file=str(bank_path.absolute()) if bank_path else None,
        bank_sha256=digest(bank_path) if bank_path else None, bank=bank, records=records,
        prompt=config['prompt'], max_new_tokens=config['max_new_tokens'], seed=42,
        decoder='fast_dllm', steps=128, block_length=32, temperature=0., revision=revision,
        vector=str(vector), vector_sha256=digest(vector), vector_info=info, extraction_ids=training,
        primary_metric=primary, own_condition=config.get('steered_condition', OWN_CONDITIONS.get(task, 'neutral-L31-answer-a1')),
        reuse_runs=[str(Path(p).absolute()) for p in entry.get('reuse_runs', [])])


def condition(strength):
    return 'neutral' if strength == 0 else f'L31-answer-a{strength:g}'


def jobs_for(datasets, kind):
    jobs = []
    for target in datasets:
        for source in datasets:
            if (kind == 'strengths') != (target == source):
                continue
            jobs.append(dict(target=target, source=source,
                directory=target if kind == 'strengths' else f'{target}__from__{source}',
                strengths=list(STRENGTHS) if kind == 'strengths' else [0., 1.]))
    return jobs


def prepare(catalog, output, kind):
    entries = json.loads(catalog.read_text())
    datasets, failures = {}, []
    for name, entry in entries.items():
        try:
            datasets[name] = load_dataset(name, entry)
        except (OSError, KeyError, ValueError) as error:
            failures.append(f'{name}: {error}')
    if failures:
        raise ValueError('Preflight failed before loading a model. Check catalog paths and original data:\n'+'\n'.join(failures))
    if len(datasets) < 2 or len({d['vector_info']['width'] for d in datasets.values()}) != 1:
        raise ValueError('Need at least two compatible DIFFA dataset vectors.')
    jobs = jobs_for(datasets, kind)
    manifest = dict(protocol=PROTOCOL, kind=kind, catalog=str(catalog.absolute()), catalog_sha256=digest(catalog),
        datasets=datasets, jobs=jobs, vector_normalization='None: preserve original raw mean-vector magnitudes.',
        selection='Existing frozen target evaluation cohorts only. No TTS, new vectors, or outcome filtering.',
        note='All strengths, transfer results, p-values and bootstrap intervals are exploratory/unadjusted.')
    freeze(output/'queue_manifest.json', manifest)
    for job in jobs:
        target, source = datasets[job['target']], datasets[job['source']]
        overlap = sorted(set(source['extraction_ids']) & {r['question_id'] for r in target['records']})
        if overlap:
            raise ValueError(f'Source extraction IDs overlap target evaluation: {job}, {overlap}')
        folder = output/job['directory']; folder.mkdir(exist_ok=True)
        plan = dict(protocol=PROTOCOL, kind=kind, source=source, target=target,
            strengths=job['strengths'], layer=31, group='answer', schedule='always', mode='fixed')
        freeze(folder/'run_config.json', plan)
        reuse_answers(plan, folder)
        done = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(folder/'results.jsonl')}
        pending = sum((r['question_id'], r['voice'], condition(a)) not in done
                      for a in job['strengths'] for r in target['records'])
        print(f'{job["directory"]}: {len(target["records"])} target questions; {pending} generations pending.', flush=True)
    return manifest


def candidate_record_map(config):
    return {(r['question_id'], r['voice']): r for r in neutral_records(config)}


def saved_strength(row, config, target):
    label = row.get('condition', row.get('style'))
    if label == 'neutral':
        return 0.
    if label == 'angry':
        return None
    interventions = row.get('interventions')
    if interventions:
        if len(interventions) != 1:
            return None
        item = interventions[0]
        if item['layer'] == 31 and item['group'] == 'answer' and item.get('schedule', 'always') == 'always':
            return float(item['strength'])
        return None
    if label == config.get('steered_condition', target['own_condition']):
        if config.get('layer', 31) == 31 and config.get('group', 'answer') == 'answer':
            return float(config.get('strength', 1.))
    return None


def reusable(row, config, record, target):
    prompt = row.get('prompt', row.get('wrapper', config.get('prompt')))
    if prompt != target['prompt']:
        return False
    for k in ('seed', 'max_new_tokens', 'decoder', 'steps', 'block_length', 'temperature'):
        if row.get(k, config.get(k, {'steps': 128, 'block_length': 32, 'temperature': 0.}.get(k))) != target[k]:
            return False
    if row.get('audio_sha256', record.get('audio_sha256')) != record['audio_sha256']:
        return False
    if row.get('audio_sha256') is None:
        previous = candidate_record_map(config).get((row['question_id'], row['voice']))
        if previous is None or previous.get('audio_sha256') != record['audio_sha256']:
            return False
        if row.get('audio_path') != previous.get('audio_path'):
            return False
    if target['task'] != 'ifeval' and str(row.get('reference_answer', '')) != record['reference_answer']:
        return False
    return True


def reuse_answers(plan, output):
    target, source = plan['target'], plan['source']
    records = {(r['question_id'], r['voice']): r for r in target['records']}
    path = output/'results.jsonl'
    done = {(r['question_id'], r['voice'], r['condition']): r for r in read_rows(path, repair=True)}
    with path.open('a') as stream:
        for folder in [Path(target['run']), *map(Path, target['reuse_runs'])]:
            if not folder.exists():
                continue
            config = json.loads(config_file(folder).read_text())
            candidate_vector = vector_path(folder, config)
            vector_matches = candidate_vector.is_file() and digest(candidate_vector) == source['vector_sha256']
            for row in read_rows(folder/'results.jsonl'):
                record = records.get((row['question_id'], row['voice']))
                a = saved_strength(row, config, target)
                if record is None or a not in plan['strengths'] or (a != 0 and not vector_matches):
                    continue
                if not reusable(row, config, record, target):
                    continue  # A different historical prompt/recipe must be regenerated, never relabeled.
                key = row['question_id'], row['voice'], condition(a)
                if key in done:
                    if done[key]['response'] != row['response']:
                        raise ValueError(f'Conflicting compatible saved responses: {key}, {folder}')
                    continue
                saved = dict(record, response=row['response'], condition=condition(a), strength=a,
                    prompt=target['prompt'], seed=42, max_new_tokens=target['max_new_tokens'],
                    decoder='fast_dllm', steps=128, block_length=32, temperature=0.,
                    source_dataset=source['name'], target_dataset=target['name'],
                    vector_sha256=source['vector_sha256'] if a else None,
                    interventions=[dict(layer=31, group='answer', strength=a)] if a else [],
                    reused_from=str(folder), audio_encoding=row.get('audio_encoding'))
                stream.write(json.dumps(saved)+'\n'); stream.flush(); done[key] = saved


def seed():
    import numpy as np
    import torch
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)


def generate_job(plan, folder, model, tokenizer, features):
    import soundfile as sf
    import torch
    from extract_vectors import gen_chunked
    target, source = plan['target'], plan['source']
    if digest(source['vector']) != source['vector_sha256']:
        raise ValueError('Source vector changed after queue preparation.')
    vector, _ = read_vector(source['vector'], source.get('emotion','angry'))
    if vector.numel() != model.llm_model.get_input_embeddings().weight.shape[1]:
        raise ValueError('Source direction does not match the DIFFA hidden dimension.')
    vector = vector.to(device='cuda', dtype=torch.bfloat16)
    layers = model.llm_model.model.transformer.blocks
    path = folder/'results.jsonl'
    done = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(path, repair=True)}
    pending = [(r, a) for a in plan['strengths'] for r in target['records']
               if (r['question_id'], r['voice'], condition(a)) not in done]
    with path.open('a') as stream:
        for i, (record, a) in enumerate(pending, 1):
            if digest(record['audio_file']) != record['audio_sha256']:
                raise ValueError('Target audio changed during the queue.')
            audio, sr = sf.read(record['audio_file'], dtype='float32')
            seed()
            interventions = [dict(layer=31, group='answer', strength=a)] if a else []
            print(f'[{i}/{len(pending)}] {source["name"]} -> {target["name"]}, x{a:g}, {record["question_id"]}', flush=True)
            started = time.monotonic()
            with install_interventions(layers, {(31, 'answer'): vector}, interventions, {}, target['max_new_tokens']) as states:
                response, encoding = gen_chunked(audio, sr, model, tokenizer, features,
                    target['prompt'], target['max_new_tokens'], 30)
            saved = dict(record, response=response, condition=condition(a), strength=a,
                prompt=target['prompt'], seed=42, max_new_tokens=target['max_new_tokens'],
                decoder='fast_dllm', steps=128, block_length=32, temperature=0.,
                source_dataset=source['name'], target_dataset=target['name'],
                vector_sha256=source['vector_sha256'] if a else None, interventions=interventions,
                steered_forwards=max(s['state']['steered'] for s in states), audio_encoding=encoding,
                generation_seconds=time.monotonic()-started, reused_from=None)
            stream.write(json.dumps(saved)+'\n'); stream.flush()


def score_rows(plan, rows, folder):
    target = plan['target']; task = target['task']; bank = target['bank']
    if task == 'ifeval':
        import run_spoken_ifeval as ifeval
        ifeval.setup_official()
        evaluator = ifeval.official_scorer()
        scored = []
        for r in rows:
            result = ifeval.score(bank[r['question_id']], r['response'], evaluator)
            metrics = {mode+'_prompt': result[mode]['prompt'] for mode in ('strict', 'loose')}
            scored.append(dict(r, metrics=metrics, official_scores=result))
        return scored, dict(repository='google-research/google-research', revision=ifeval.REVISION,
            adaptation='Original official checkers on our spoken adaptation; no factual correctness metric.')
    from rescore_audio_steering import HarnessTaskScorer, numeric_equal, numeric_within_tolerance
    if task in ('bbh3', 'bbh5'):
        from score_bbh_harness import HarnessScorer
        scorer = HarnessScorer(folder/'harness/upstream', objects='three' if task == 'bbh3' else 'five')
        docs = [dict(input=bank[r['question_id']]['original_input'], target=bank[r['question_id']]['original_target']) for r in rows]
    else:
        scorer = HarnessTaskScorer('gsm8k', folder/'harness/upstream')
        docs = [dict(question='Saved spoken math question', answer=r['reference_answer']) for r in rows]
    values = scorer.score([r['response'] for r in rows], docs)
    scored = []
    for i, r in enumerate(rows):
        metrics = {m: v[i]['correct'] for m, v in values.items()}
        predictions = {m: v[i]['prediction'] for m, v in values.items()}
        if task not in ('bbh3', 'bbh5'):
            metrics['flexible-numeric'] = numeric_equal(predictions['flexible-extract'], r['reference_answer'])
            metrics['format-compliant'] = predictions['strict-match'] != '[invalid]'
            if task == 'svamp':
                metrics['author-tolerance'] = numeric_within_tolerance(predictions['flexible-extract'], r['reference_answer'])
        scored.append(dict(r, metrics=metrics, predictions=predictions))
    return scored, scorer.provenance


def report_job(folder):
    from rescore_audio_steering import paired
    if not (folder/'run_config.json').exists():
        return []
    plan = json.loads((folder/'run_config.json').read_text())
    rows = read_rows(folder/'results.jsonl')
    if not rows:
        return []
    scored, provenance = score_rows(plan, rows, folder)
    baseline = {(r['question_id'], r['voice']): r for r in scored if r['strength'] == 0}
    results = []
    print(f'\n{plan["source"]["name"]} -> {plan["target"]["name"]}; target prompt and budget preserved.')
    for a in plan['strengths']:
        group = {(r['question_id'], r['voice']): r for r in scored if r['strength'] == a}
        keys = sorted(group.keys() & baseline.keys())
        if not keys:
            continue
        for metric in next(iter(group.values()))['metrics']:
            result = paired([(baseline[k]['metrics'][metric], group[k]['metrics'][metric]) for k in keys])
            results.append(dict(source=plan['source']['name'], target=plan['target']['name'], strength=a,
                                metric=metric, **result))
            print(f'{metric:18} x{a:4g} N={len(keys):4d} '
                  f'{result["baseline_correct"]/len(keys):6.1%} -> {result["steered_correct"]/len(keys):6.1%}; '
                  f'gain={result["gain_pp"]:+.2f}; W/L={result["wins"]}/{result["losses"]}; '
                  f'p={result["mcnemar_exact_p"]:.4f}; CI={result["ci95_gain_pp"]}')
    instruction_totals = []
    if plan['target']['task'] == 'ifeval':
        for a in plan['strengths']:
            group = [r for r in scored if r['strength'] == a]
            for mode in ('strict', 'loose'):
                bits = [b for r in group for b in r['official_scores'][mode]['instructions']]
                if bits:
                    instruction_totals.append(dict(strength=a, mode=mode, instructions=len(bits), correct=sum(bits), accuracy=sum(bits)/len(bits)))
    atomic_text(folder/'rescored.jsonl', ''.join(json.dumps(r)+'\n' for r in scored))
    atomic_text(folder/'summary.json', json.dumps(dict(scorer=provenance, primary_metric=plan['target']['primary_metric'],
        paired=results, instruction_totals=instruction_totals,
        note='Exploratory; unadjusted exact McNemar tests and 95% question-paired bootstrap intervals. '
             'Numeric equivalence/format compliance are additional diagnostics; no cross-task metric pooling.'), indent=2)+'\n')
    return results


def combined_report(output, manifest, rescore=True):
    combined = []
    for job in manifest['jobs']:
        folder = output/job['directory']
        if rescore:
            combined.extend(report_job(folder))
        elif (folder/'summary.json').exists():
            combined.extend(json.loads((folder/'summary.json').read_text())['paired'])
    primary = [r for r in combined if r['metric'] == manifest['datasets'][r['target']]['primary_metric']
               and (manifest['kind'] == 'strengths' or r['strength'] != 0)]
    path = output/('strength-summary.csv' if manifest['kind'] == 'strengths' else 'transfer-matrix.csv')
    with path.open('w', newline='') as stream:
        fields = ['source', 'target', 'strength', 'metric', 'pairs', 'baseline_correct', 'steered_correct',
                  'gain_pp', 'wins', 'losses', 'mcnemar_exact_p', 'ci95_gain_pp']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(primary)
    atomic_text(output/'summary.json', json.dumps(dict(primary_results=primary, all_metrics=combined,
        note=manifest['note']), indent=2)+'\n')
    print('Saved', path, flush=True)


def run_queue(output, manifest):
    # The one shared lock serializes these two queues and the original experiments.
    with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
        print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            print('Waiting for active GPU work.', flush=True); time.sleep(30)
        cache = Path(os.environ.get('HF_HOME') or next((str(p) for p in
            (Path('/workspace/.hf_home'), Path('/workspace/hf-cache')) if (p/'hub').is_dir()), 'outputs/hf-diffa')).absolute()
        os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache/'hub'), HUGGINGFACE_HUB_CACHE=str(cache/'hub'))
        os.environ.pop('TRANSFORMERS_CACHE', None)
        model = tokenizer = features = None
        for i, job in enumerate(manifest['jobs'], 1):
            folder = output/job['directory']; plan = json.loads((folder/'run_config.json').read_text())
            done = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(folder/'results.jsonl')}
            pending = any((r['question_id'], r['voice'], condition(a)) not in done
                          for a in plan['strengths'] for r in plan['target']['records'])
            print(f'STAGE [{i}/{len(manifest["jobs"])}]: {job["directory"]}', flush=True)
            if pending:
                if model is None:
                    import torch
                    from extract_vectors import load_model
                    model, tokenizer, features = load_model()
                    (output/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__, transformers=version('transformers'),
                        extract_vectors_sha256=digest(Path('extract_vectors.py')), steer_sha256=digest(Path('steer.py'))), indent=2)+'\n')
                generate_job(plan, folder, model, tokenizer, features)
            report_job(folder)
            combined_report(output, manifest, rescore=False)


def main(kind):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['all', 'prepare', 'summary'], default='all')
    p.add_argument('--catalog', type=Path, default=Path('winning_diffa_catalog.json'))
    p.add_argument('--output', type=Path, default=Path('outputs/winning-'+kind))
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        manifest = args.output/'queue_manifest.json'
        if not manifest.exists():
            print('Queue not prepared yet; it may still be waiting for original input jobs. Check its log.')
            return
        combined_report(args.output, json.loads(manifest.read_text()))
        return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare(args.catalog, args.output, kind)
        total = sum(len(manifest['datasets'][j['target']]['records'])*len(j['strengths']) for j in manifest['jobs'])
        print(f'{len(manifest["jobs"])} jobs; {total} planned answers including reusable baselines.', flush=True)
        if args.mode != 'prepare':
            run_queue(args.output, manifest)
