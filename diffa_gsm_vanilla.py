"""Vanilla GSM natural deliveries, random/agreed fitting, then held-out x+1 steering."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

import diffa_happy_nightly as task
from jenny_bundle_registry import Registry, STYLES
from steer import read_rows

EMOTIONS = ('angry', 'happy', 'sad')
REGIMES = ('random', 'agreed')
PROMPT = ('Solve the math problem in the audio. End your response with a final answer line in the format:\n'
          '#### <number>')
PROTOCOL = 'diffa-gsm-vanilla-random-agreed-v1'
# Verified against the uploaded Jenny ZIP and agreed-runs metadata, before generation.
EXPECTED_SPLIT_SHA256 = 'e02dabe85aacd64ab4b9595959aed518eac33cde602bf54eeec3253ab8755873'
RECIPE = dict(seed=42, max_new_tokens=512, steps=128, block_length=32, temperature=0., decoder='fast_dllm')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def select(pool, agreed, test_count=500, fit_count=425):
    """Choose evaluation without labels; fit each emotion independently, outside evaluation."""
    if len(set(pool)) != len(pool):
        raise ValueError('Duplicate question IDs.')
    order = sorted(pool)
    random.Random(42).shuffle(order)
    test, fitting = order[:test_count], order[test_count:]
    if len(test) != test_count or len(fitting) < fit_count:
        raise ValueError('Insufficient complete audio questions.')
    common = set(random.Random(43).sample(sorted(fitting), fit_count))
    selected = {}
    for emotion in EMOTIONS:
        eligible = sorted(set(fitting) & set(agreed[emotion]))
        if len(eligible) < fit_count:
            raise ValueError(f'{emotion}: only {len(eligible)} recognized fitting questions; need {fit_count}.')
        recognized = set(random.Random(44).sample(eligible, fit_count))
        # A common traversal makes both aggregate checkpoints independently resumable prefixes.
        selected[emotion] = dict(random=[q for q in fitting if q in common],
                                 agreed=[q for q in fitting if q in recognized])
    return dict(test_ids=test, fitting_order=fitting, collections=selected)


def prepare(registry, output, expected_pool=1319, test_count=500, fit_count=425):
    bank, records, labels = {}, {}, {}
    for q in registry.neutral_ids('gsm8k'):
        neutral = registry.choose('gsm8k', q, 'neutral')
        entries = {s: registry.choose('gsm8k', q, s, question=neutral['question'], answer=neutral['answer'])
                   for s in STYLES}
        if not all(entries.values()):
            continue
        for s, item in entries.items():
            variants = {x['record']['audio_sha256'] for x in registry.options['gsm8k', q, s]
                        if x['question'] == neutral['question'] and x['answer'] == neutral['answer']}
            if len(variants) != 1:
                raise ValueError(f'Ambiguous audio: {q}/{s}. Pin the same Jenny ZIP on both machines.')
            if s != 'neutral':
                label = registry.labels.get(item['record']['audio_sha256'], {})
                if label.get('classification_status') != 'classified':
                    raise ValueError(f'Missing saved emotion2vec classification: {q}/{s}.')
                labels[q, s] = label['emotion']
            records[q, s] = item['record']
        bank[q] = dict(question_id=q, question=neutral['question'], answer=neutral['answer'])
    if len(bank) != expected_pool:
        raise ValueError(f'Expected {expected_pool} complete-four GSM questions; found {len(bank)}. Restore the complete Jenny ZIP.')
    agreed = {e: [q for q in bank if labels[q, e] == e] for e in EMOTIONS}
    split = select(list(bank), agreed, test_count, fit_count)
    manifest = dict(protocol=PROTOCOL, prompt=PROMPT, recipe=RECIPE, **split,
                    bank=bank, audio_sha256={q: {s: records[q, s]['audio_sha256'] for s in STYLES} for q in sorted(bank)},
                    emotion2vec_predictions={q: {e: labels[q, e] for e in EMOTIONS} for q in sorted(bank)},
                    selection='Sorted complete-four pool; shuffle seed42 and reserve first500 BEFORE labels. '
                              'Random425 seed43; emotion-specific agreed425 seed44. Neutral prediction ignored. '
                              'No outcome filtering. All fitting excludes the common evaluation set.',
                    seeds=dict(evaluation=42, random_collection=43, agreed_collection=44),
                    counts=dict(pool=expected_pool, evaluation=test_count, fitting_per_regime=fit_count))
    checksum = fingerprint(manifest)
    if (expected_pool, test_count, fit_count) == (1319, 500, 425) and checksum != EXPECTED_SPLIT_SHA256:
        raise ValueError(f'Jenny metadata differs from the audited bundle: split SHA256 {checksum}; '
                         f'expected {EXPECTED_SPLIT_SHA256}. No generation started; restore the same audio/label bundle.')
    output.mkdir(parents=True, exist_ok=True)
    task.winners.freeze(output/'split.json', manifest)
    print(f'Vanilla split SHA256: {checksum}', flush=True)
    print(f'{len(bank)} natural-delivery questions; {fit_count} random + {fit_count} agreed fitting questions per emotion.', flush=True)
    for e in EMOTIONS:
        fitting = set(split['fitting_order'])
        common = set(split['collections'][e]['random'])
        print(f'{e}: {len(fitting & set(agreed[e]))} recognized outside evaluation; '
              f'random/agreed fitting overlap={len(common & set(split["collections"][e]["agreed"]))}.', flush=True)
    plans = {}
    for e in EMOTIONS:
        plans[e] = {}
        for regime in REGIMES:
            train, test = split['collections'][e][regime], split['test_ids']
            target = dict(name='gsm8k', task='gsm8k', prompt=PROMPT, **RECIPE,
                          records=[records[q, 'neutral'] for q in test], bank={q: bank[q] for q in test}, extraction_ids=train)
            plan = dict(protocol=PROTOCOL, split_sha256=checksum, name='gsm8k', emotion=e, regime=regime,
                        prompt=PROMPT, collection_prompt=PROMPT, **RECIPE, bank=bank, target=target,
                        collection_ids=train, test_ids=test,
                        pairs=[dict(question_id=q, neutral=records[q, 'neutral'], emotion=records[q, e]) for q in train],
                        recordings=[dict(records[q, s], condition=s) for q in test for s in ('neutral', e)],
                        averaging='Matched forwards within each pair; token-position means; equal per-question weight.')
            folder = output/e/regime
            folder.mkdir(parents=True, exist_ok=True)
            task.winners.freeze(folder/'run_config.json', plan)
            plans[e][regime] = plan
    return manifest, plans


def prepare_deliveries(registry, manifest, plans, output):
    """Extend natural deliveries using a separate manifest; never rewrite fitting configs."""
    ids = manifest.get('delivery_ids', manifest.get('test_ids', []) + manifest['fitting_order'])
    base = plans['angry']['random']
    prompt = base['prompt']
    recipe = {k: base[k] for k in RECIPE}
    records = []
    for q in ids:
        example = manifest['bank'][q]
        for style in STYLES:
            item = registry.choose(base['name'], q, style, question=example['question'], answer=example['answer'],
                                   expected=manifest['audio_sha256'][q][style])
            if item is None:
                raise ValueError(f'Missing pinned full-delivery audio: {q}/{style}.')
            records.append(dict(item['record'], condition=style))
    config = dict(protocol=manifest['protocol']+'-all-deliveries-v1', split_sha256=fingerprint(manifest),
                  prompt=prompt, **recipe, question_ids=ids, steering_holdout_ids=manifest.get('test_ids', []),
                  bank=manifest['bank'], recordings=records,
                  note='All natural deliveries, including fitting questions. No steering or outcome-based selection.')
    folder = output/'delivery'
    folder.mkdir(exist_ok=True)
    task.winners.freeze(folder/'run_config.json', config)
    return delivery_scope(plans, output)


def delivery_scope(plans, output):
    path = output/'delivery'/'run_config.json'
    if not path.exists():
        return plans  # Read earlier 500-only runs before their next launch.
    config = json.loads(path.read_text())
    base = plans['angry']['random']
    if (config['split_sha256'] != base['split_sha256'] or config['prompt'] != base['prompt']
            or any(config[k] != base[k] for k in RECIPE)
            or config['steering_holdout_ids'] != base['test_ids']
            or set(config['question_ids']) != set(base['bank'])
            or len(config['question_ids']) != len(set(config['question_ids']))):
        raise ValueError('Full-delivery manifest differs from the frozen vanilla split.')
    expected = {(q, s) for q in config['question_ids'] for s in STYLES}
    actual = {(r['question_id'], r['condition']) for r in config['recordings']}
    if actual != expected or len(actual) != len(config['recordings']):
        raise ValueError('Full-delivery manifest must contain each question/style exactly once.')
    result = {}
    for e in EMOTIONS:
        result[e] = {}
        for regime, plan in plans[e].items():
            result[e][regime] = dict(plan, test_ids=config['question_ids'],
                target=dict(plan['target'], bank=config['bank'],
                            records=[r for r in config['recordings'] if r['condition'] == 'neutral']),
                recordings=[r for r in config['recordings'] if r['condition'] in ('neutral', e)])
    return result


def delivery_rows(plans, folder, repair=False):
    expected = {(r['question_id'], r['condition']): r for p in plans.values() for r in p['random']['recordings']}
    result = {}
    for row in read_rows(folder/'results.jsonl', repair=repair):
        key = row['question_id'], row['condition']
        if (key not in expected or key in result or row.get('voice') != expected[key]['voice']
                or row.get('interventions') != [] or row.get('strength') != 0.
                or not task.winners.reusable(row, {}, expected[key], plans['angry']['random']['target'])):
            raise ValueError('Vanilla delivery cache differs from frozen audio/prompt/recipe, or contains duplicates/steering.')
        result[key] = row
    return result


def deliver(plans, emotions, folder, model, tokenizer, features, question_ids=None):
    folder.mkdir(exist_ok=True)
    done = delivery_rows(plans, folder, repair=True)
    base = plans['angry']['random']
    recipe = {k: base[k] for k in RECIPE}
    records = {}
    for e in emotions:
        for r in plans[e]['random']['recordings']:
            if question_ids is not None and r['question_id'] not in question_ids:
                continue
            records[r['question_id'], r['condition']] = r
    with (folder/'results.jsonl').open('a') as stream:
        for key, record in records.items():
            if key in done:
                continue
            print(f'Vanilla delivery {key[1]} {key[0]} [{len(done)+1}]', flush=True)
            response, encoding = task.generate(record, base, base['prompt'], model, tokenizer, features)
            row = dict(record, response=response, prompt=base['prompt'], **recipe, strength=0., interventions=[],
                       audio_encoding=encoding, response_tokens=len(tokenizer.encode(response, add_special_tokens=False)))
            stream.write(json.dumps(row)+'\n')
            stream.flush()
            done[key] = row


def reuse_collection_deliveries(plans, scoped, emotions, output, tokenizer):
    """Collection only observes native generation; reuse those exact natural answers."""
    from diffa_collection_report import collection_rows
    folder = output/'delivery'
    done = delivery_rows(scoped, folder, repair=True)
    expected = {(r['question_id'], r['condition']): r for e in emotions for r in scoped[e]['random']['recordings']}
    copied = 0
    with (folder/'results.jsonl').open('a') as stream:
        for e in emotions:
            for regime in REGIMES:
                source = output/e/regime
                for row in collection_rows(plans[e][regime], source):
                    key = row['question_id'], row['condition']
                    if (key not in expected or row.get('strength') != 0. or row.get('interventions') != []
                            or not task.winners.reusable(row, {}, expected[key], scoped[e]['random']['target'])):
                        raise ValueError('Collection answer cannot be reused as the same vanilla natural delivery.')
                    if key in done:
                        continue
                    row = dict(row, response_tokens=len(tokenizer.encode(row['response'], add_special_tokens=False)),
                               reused_from=str(source), split='all-natural-deliveries')
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                    done[key] = row
                    copied += 1
    print(f'Reused {copied} already-generated collection deliveries; no inference repeated.', flush=True)


def collect(plans, emotions, order, output, model, tokenizer, features):
    """Capture each pending question/style once; reuse overlap without full activation caches."""
    import torch
    import diffa_stream_collection as running
    import src.fast_dllm.diffa_generate as decoder
    jobs = {(e, r): plans[e][r] for e in emotions for r in REGIMES}
    states = {k: running.load(p, output/k[0]/k[1], task.winners.digest(output/k[0]/k[1]/'run_config.json'))
              for k, p in jobs.items()}
    pairs = {k: {p['question_id']: p for p in plan['pairs']} for k, plan in jobs.items()}
    for q in order:
        pending = [k for k in jobs if q in pairs[k] and q not in states[k]['completed_ids']]
        if not pending:
            continue
        for k in pending:
            if jobs[k]['collection_ids'][len(states[k]['completed_ids'])] != q:
                raise ValueError('Collection traversal differs from checkpoint prefix.')
        base = jobs[pending[0]]
        record = pairs[pending[0]][q]['neutral']
        print(f'Vanilla fitting {q}: {pending}', flush=True)
        nr, ne, n = task.emotion.capture(model, lambda: task.generate(record, base, PROMPT, model, tokenizer, features), decoder, 512)
        for e in emotions:
            keys = [k for k in pending if k[0] == e]
            if not keys:
                continue
            record = pairs[keys[0]][q]['emotion']
            response, encoding, h = task.emotion.capture(model, lambda: task.generate(record, base, PROMPT, model, tokenizer, features), decoder, 512)
            nm, hm, delta, matched = task.average_pair(n, h, 'matched')
            saved = dict(question_id=q, neutral=nm, emotion=hm, difference=delta, neutral_response=nr,
                         emotion_response=response, neutral_forwards=n.shape[1], emotion_forwards=h.shape[1],
                         matched_forwards=matched, neutral_encoding=ne, emotion_encoding=encoding)
            for k in keys:
                running.append(states[k], saved)
                task.save_tensor(torch, states[k], output/k[0]/k[1]/'collection-stream.pt')
            del h, nm, hm, delta, saved
        del n
    for k, plan in jobs.items():
        running.finalize(plan, output/k[0]/k[1], states[k])


def generate_and_collect(plans, scoped, emotions, order, output, model, tokenizer, features):
    """One question pass: save natural responses and observe pending fitting pairs together."""
    import torch
    import diffa_stream_collection as running
    import src.fast_dllm.diffa_generate as decoder
    reuse_collection_deliveries(plans, scoped, emotions, output, tokenizer)
    natural = delivery_rows(scoped, output/'delivery', repair=True)
    records = {(r['question_id'], r['condition']): r for e in emotions for r in scoped[e]['random']['recordings']}
    jobs = {(e, r): plans[e][r] for e in emotions for r in REGIMES}
    states = {k: running.load(p, output/k[0]/k[1], task.winners.digest(output/k[0]/k[1]/'run_config.json'))
              for k, p in jobs.items()}
    membership = {k: set(p['collection_ids']) for k, p in jobs.items()}
    base = plans[emotions[0]]['random']
    prompt = base['prompt']
    recipe = {k: base[k] for k in RECIPE}
    budget = base['max_new_tokens']
    with (output/'delivery/results.jsonl').open('a') as stream:
        def save(q, style, response, encoding):
            key = q, style
            if key in natural:
                return
            row = dict(records[key], response=response, prompt=prompt, **recipe, strength=0., interventions=[],
                       audio_encoding=encoding, response_tokens=len(tokenizer.encode(response, add_special_tokens=False)))
            stream.write(json.dumps(row)+'\n'); stream.flush()
            natural[key] = row

        for index, q in enumerate(order):
            pending = [k for k in jobs if q in membership[k] and q not in states[k]['completed_ids']]
            missing = [s for s in ('neutral', *emotions) if (q, s) not in natural]
            if not pending and not missing:
                continue
            for k in pending:
                if jobs[k]['collection_ids'][len(states[k]['completed_ids'])] != q:
                    raise ValueError('Question traversal differs from the frozen collection checkpoint.')
            print(f'{base["name"]} [{index+1}/{len(order)}] {q}: missing deliveries={missing}; capture for={pending}', flush=True)
            n = None
            if pending:
                nr, ne, n = task.emotion.capture(model,
                    lambda: task.generate(records[q, 'neutral'], base, prompt, model, tokenizer, features), decoder, budget)
                save(q, 'neutral', nr, ne)
            elif 'neutral' in missing:
                response, encoding = task.generate(records[q, 'neutral'], base, prompt, model, tokenizer, features)
                save(q, 'neutral', response, encoding)
            for e in emotions:
                keys = [k for k in pending if k[0] == e]
                if keys:
                    response, encoding, h = task.emotion.capture(model,
                        lambda: task.generate(records[q, e], base, prompt, model, tokenizer, features), decoder, budget)
                    save(q, e, response, encoding)
                    nm, hm, delta, matched = task.average_pair(n, h, 'matched')
                    saved = dict(question_id=q, neutral=nm, emotion=hm, difference=delta,
                        neutral_response=nr, emotion_response=response, neutral_forwards=n.shape[1],
                        emotion_forwards=h.shape[1], matched_forwards=matched, neutral_encoding=ne, emotion_encoding=encoding)
                    for k in keys:
                        running.append(states[k], saved)
                        task.save_tensor(torch, states[k], output/k[0]/k[1]/'collection-stream.pt')
                    del h, nm, hm, delta, saved
                elif e in missing:
                    response, encoding = task.generate(records[q, e], base, prompt, model, tokenizer, features)
                    save(q, e, response, encoding)
            del n
    for k, plan in jobs.items():
        running.finalize(plan, output/k[0]/k[1], states[k])


def steering_job(plan, folder, all_deliveries=None):
    source = task.descriptor(plan, folder)
    source.setdefault('primary_metric', 'flexible-numeric')
    target = source
    if all_deliveries is not None:
        excluded = set(plan['collection_ids'])
        records = [r for r in all_deliveries['target']['records'] if r['question_id'] not in excluded]
        target = dict(source, records=records, bank={r['question_id']: plan['bank'][r['question_id']] for r in records})
        if len(records) != len(plan['bank'])-len(excluded):
            raise ValueError('Steering must include every question outside this vector collection.')
    if set(plan['collection_ids']) & {r['question_id'] for r in target['records']}:
        raise ValueError('Fitting questions overlap steering evaluation.')
    return dict(protocol=PROTOCOL+'-steering-v1', kind='vanilla-heldout-steering',
                split_sha256=plan['split_sha256'], emotion=plan['emotion'], regime=plan['regime'],
                source=source, target=target, strengths=[0., 1.],
                layer=31, group='answer', schedule='always', mode='fixed')


def steering_rows(job, folder, repair=False):
    expected = {(r['question_id'], r['voice']): r for r in job['target']['records']}
    result = {}
    for row in read_rows(folder/'results.jsonl', repair=repair):
        strength = row.get('strength')
        key = row['question_id'], row['voice'], row['condition']
        record = expected.get(key[:2])
        interventions = [dict(layer=31, group='answer', strength=1.)] if strength == 1. else []
        if (record is None or strength not in (0., 1.) or key in result
                or row['condition'] != task.winners.condition(strength)
                or row.get('interventions') != interventions
                or row.get('vector_sha256') != (job['source']['vector_sha256'] if strength == 1. else None)
                or not task.winners.reusable(row, {}, record, job['target'])):
            raise ValueError('Steering cache differs from the exact holdout, vector, prompt or strength.')
        result[key] = row
    return result


def steer(plans, scoped, emotions, output, model, tokenizer, features):
    natural = delivery_rows(scoped, output/'delivery')
    for e in emotions:
        for regime in REGIMES:
            parent = output/e/regime
            job = steering_job(plans[e][regime], parent, scoped[e][regime])
            folder = parent/'strength-1-unfitted'
            folder.mkdir(exist_ok=True)
            task.winners.freeze(folder/'run_config.json', job)
            saved = steering_rows(job, folder, repair=True)
            with (folder/'results.jsonl').open('a') as stream:
                for record in job['target']['records']:
                    key = record['question_id'], record['voice'], 'neutral'
                    if key in saved:
                        continue
                    row = natural.get((record['question_id'], 'neutral'))
                    if row is None:
                        raise ValueError('Complete natural neutral deliveries before steering.')
                    copied = dict(row, reused_from=str(output/'delivery'), vector_sha256=None)
                    stream.write(json.dumps(copied)+'\n')
                    stream.flush()
            # All x0 baselines are present, so generate_job only runs x+1 interventions.
            print(f'{e}/{regime}: L31 answer x+1, {len(job["target"]["records"])} held-out neutrals.', flush=True)
            task.winners.generate_job(job, folder, model, tokenizer, features)
            steering_rows(job, folder)
            task.report_steering(folder, tokenizer)


def report(plans, output, emotions, tokenizer=None, detailed=False):
    from rescore_audio_steering import paired
    scoped = delivery_scope(plans, output)
    raw = list(delivery_rows(scoped, output/'delivery').values())
    scored_groups = {}
    for e in emotions:
        plan = scoped[e]['random']
        folder = output/e/'delivery-report'
        folder.mkdir(parents=True, exist_ok=True)
        rows = [r for r in raw if r['condition'] in ('neutral', e)]
        print(f'{e}: ALL natural deliveries ({len(plan["test_ids"])} questions; includes fitting questions)', flush=True)
        task.report_delivery(plan, folder, tokenizer, rows=rows, cohort='all natural deliveries')
        summary = folder/'summary.json'
        if summary.exists():
            data = json.loads(summary.read_text())
            data.update(split_sha256=plan['split_sha256'],
                        note='All natural vanilla deliveries, including fitting questions; no historical answers or steering. '
                             'Exploratory unadjusted paired tests. Collection responses are reported separately.')
            task.winners.atomic_text(summary, json.dumps(data, indent=2)+'\n')
        path = folder/'rescored.jsonl'
        if path.exists():
            scored_groups[e] = {r['question_id']: r for r in read_rows(path) if r['condition'] == e}
        for regime in REGIMES:
            folder = output/e/regime
            import diffa_stream_collection as running
            state = running.load(plans[e][regime], folder, task.winners.digest(folder/'run_config.json'))
            print(f'{e}/{regime}: vector collection {len(state["completed_ids"])}/{len(plans[e][regime]["collection_ids"])}.', flush=True)
            if detailed:
                task.report_collection(plans[e][regime], folder, tokenizer)
            steering = folder/'strength-1-unfitted'
            if not (steering/'run_config.json').exists():
                steering = folder/'strength-1'  # Preserve access to earlier steering results.
            if (steering/'run_config.json').exists():
                print(f'{e}/{regime}: HELD-OUT L31 answer x+1 steering', flush=True)
                job = json.loads((steering/'run_config.json').read_text())
                steering_rows(job, steering)
                task.report_steering(steering, tokenizer)
            else:
                print(f'{e}/{regime}: x+1 steering not started yet.', flush=True)
    comparisons = []
    for i, a in enumerate(emotions):
        for b in emotions[i+1:]:
            left, right = scored_groups.get(a, {}), scored_groups.get(b, {})
            keys = sorted(left.keys() & right.keys())
            if not keys:
                continue
            for metric in left[keys[0]]['metrics']:
                result = paired([(left[q]['metrics'][metric], right[q]['metrics'][metric]) for q in keys])
                comparisons.append(dict(reference=a, test=b, metric=metric, cohort='all natural deliveries', **result))
                print(f'{b} vs {a} / {metric}: N={len(keys)}; gain={result["gain_pp"]:+.2f} pp; p={result["mcnemar_exact_p"]:.4g}', flush=True)
    task.winners.atomic_text(output/'emotion-comparisons.json', json.dumps(comparisons, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'prepare', 'summary'), default='run')
    parser.add_argument('--emotions', nargs='+', choices=EMOTIONS, default=list(EMOTIONS))
    parser.add_argument('--audio-root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('outputs/diffa-gsm-vanilla'))
    parser.add_argument('--detailed', action='store_true', help='Also print fitting-set accuracy diagnostics.')
    args = parser.parse_args()
    args.emotions = list(dict.fromkeys(args.emotions))
    if args.mode == 'summary':
        plans = {e: {r: json.loads((args.output/e/r/'run_config.json').read_text()) for r in REGIMES} for e in EMOTIONS}
        manifest = json.loads((args.output/'split.json').read_text())
        print('Vanilla split SHA256:', fingerprint(manifest), flush=True)
        report(plans, args.output, args.emotions, detailed=args.detailed)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'runner.lock').open('a') as own:
        print('Waiting for any existing vanilla runner to finish; completed work will be reused.', flush=True)
        fcntl.flock(own, fcntl.LOCK_EX)
        print('CPU preflight: verify Jenny audio and saved e2v labels; no regeneration/classification.', flush=True)
        registry = Registry(args.audio_root)
        manifest, plans = prepare(registry, args.output)
        scoped = prepare_deliveries(registry, manifest, plans, args.output)
        if args.mode == 'prepare':
            return
        # Resolve scoring dependencies before spending GPU time on generation.
        scoring = args.output/'delivery'/'scoring-preflight'
        scoring.mkdir(parents=True, exist_ok=True)
        task.winners.score_rows(dict(target=plans['angry']['random']['target']), [], scoring)
        if shutil.disk_usage(args.output).free < 1024**3:
            raise RuntimeError('Need at least 1 GiB free for aggregate checkpoints and reports.')
        with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
            print('Waiting for DIFFA GPU lock and idle GPU; safe to restart this command.', flush=True)
            fcntl.flock(gpu, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                time.sleep(30)
            from extract_vectors import load_model
            from diffa_model_receipts import model_revisions
            model, tokenizer, features = load_model()
            task.winners.atomic_text(args.output/'runtime.json', json.dumps(dict(split_sha256=fingerprint(manifest),
                runner_sha256=task.winners.digest(__file__), model_revisions=model_revisions(),
                generator_sha256=task.winners.digest(Path(__file__).with_name('extract_vectors.py')),
                model_config=model.config.to_dict()), indent=2, default=str)+'\n')
            generate_and_collect(plans, scoped, args.emotions, manifest['test_ids']+manifest['fitting_order'],
                                 args.output, model, tokenizer, features)
            report(plans, args.output, args.emotions, tokenizer, detailed=args.detailed)
            steer(plans, scoped, args.emotions, args.output, model, tokenizer, features)
            print('DONE: all natural deliveries, both fitting regimes and both held-out x+1 steering runs.', flush=True)


if __name__ == '__main__':
    main()
