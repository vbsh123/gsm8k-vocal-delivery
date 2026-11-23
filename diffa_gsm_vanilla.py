"""Frozen vanilla GSM delivery evaluation and random/agreed vector fitting. No steering."""
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
    print(f'{len(bank)} complete questions; {test_count} COMMON evaluation; {fit_count} random + {fit_count} agreed per emotion. NO STEERING.', flush=True)
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


def deliver(plans, emotions, folder, model, tokenizer, features):
    folder.mkdir(exist_ok=True)
    done = delivery_rows(plans, folder, repair=True)
    records = {}
    for e in emotions:
        for r in plans[e]['random']['recordings']:
            records[r['question_id'], r['condition']] = r
    with (folder/'results.jsonl').open('a') as stream:
        for key, record in records.items():
            if key in done:
                continue
            print(f'Vanilla delivery {key[1]} {key[0]} [{len(done)+1}]', flush=True)
            response, encoding = task.generate(record, plans['angry']['random'], PROMPT, model, tokenizer, features)
            row = dict(record, response=response, prompt=PROMPT, **RECIPE, strength=0., interventions=[],
                       audio_encoding=encoding, response_tokens=len(tokenizer.encode(response, add_special_tokens=False)))
            stream.write(json.dumps(row)+'\n')
            stream.flush()
            done[key] = row


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


def report(plans, output, emotions, tokenizer=None):
    from rescore_audio_steering import paired
    raw = list(delivery_rows(plans, output/'delivery').values())
    scored_groups = {}
    for e in emotions:
        plan = plans[e]['random']
        folder = output/e/'delivery-report'
        folder.mkdir(parents=True, exist_ok=True)
        rows = [r for r in raw if r['condition'] in ('neutral', e)]
        task.report_delivery(plan, folder, tokenizer, rows=rows)
        summary = folder/'summary.json'
        if summary.exists():
            data = json.loads(summary.read_text())
            data.update(split_sha256=plan['split_sha256'],
                        note='New frozen vanilla GSM evaluation cohort; no historical answers or steering. '
                             'Exploratory unadjusted paired tests. Collection responses are reported separately.')
            task.winners.atomic_text(summary, json.dumps(data, indent=2)+'\n')
        path = folder/'rescored.jsonl'
        if path.exists():
            scored_groups[e] = {r['question_id']: r for r in read_rows(path) if r['condition'] == e}
        for regime in REGIMES:
            folder = output/e/regime
            print(f'{e}/{regime}: descriptive fitting results (not evaluation)', flush=True)
            task.report_collection(plans[e][regime], folder, tokenizer)
    comparisons = []
    for i, a in enumerate(emotions):
        for b in emotions[i+1:]:
            left, right = scored_groups.get(a, {}), scored_groups.get(b, {})
            keys = sorted(left.keys() & right.keys())
            if not keys:
                continue
            for metric in left[keys[0]]['metrics']:
                result = paired([(left[q]['metrics'][metric], right[q]['metrics'][metric]) for q in keys])
                comparisons.append(dict(reference=a, test=b, metric=metric, **result))
                print(f'{b} vs {a} / {metric}: N={len(keys)}; gain={result["gain_pp"]:+.2f} pp; p={result["mcnemar_exact_p"]:.4g}', flush=True)
    task.winners.atomic_text(output/'emotion-comparisons.json', json.dumps(comparisons, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'prepare', 'summary'), default='run')
    parser.add_argument('--emotions', nargs='+', choices=EMOTIONS, default=list(EMOTIONS))
    parser.add_argument('--audio-root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('outputs/diffa-gsm-vanilla'))
    args = parser.parse_args()
    args.emotions = list(dict.fromkeys(args.emotions))
    if args.mode == 'summary':
        plans = {e: {r: json.loads((args.output/e/r/'run_config.json').read_text()) for r in REGIMES} for e in EMOTIONS}
        manifest = json.loads((args.output/'split.json').read_text())
        print('Vanilla split SHA256:', fingerprint(manifest), flush=True)
        report(plans, args.output, args.emotions)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'runner.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('CPU preflight: verify Jenny audio and saved e2v labels; no regeneration/classification.', flush=True)
        manifest, plans = prepare(Registry(args.audio_root), args.output)
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
            deliver(plans, args.emotions, args.output/'delivery', model, tokenizer, features)
            report(plans, args.output, args.emotions, tokenizer)
            collect(plans, args.emotions, manifest['fitting_order'], args.output, model, tokenizer, features)
            report(plans, args.output, args.emotions, tokenizer)
            print('DONE: vanilla delivery evaluation + both fitting regimes. No steering started.', flush=True)


if __name__ == '__main__':
    main()
