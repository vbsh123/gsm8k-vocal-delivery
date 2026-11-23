"""All-dataset vanilla deliveries + simultaneous random/agreed fitting + x+1 steering."""
import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

import diffa_gsm_vanilla as engine
from jenny_bundle_registry import Registry, STYLES
from steer import read_rows

DATASETS = ('gsm8k', 'svamp', 'multiarith', 'bbh3', 'bbh5', 'ifeval')
NEXT = DATASETS[1:]
SPECS = dict(svamp=(1000, 500), multiarith=(580, 290), bbh3=(250, 125), bbh5=(250, 125), ifeval=(541, 270))
PROTOCOL = 'diffa-all-vanilla-random-agreed-v1'
IFEVAL_PROMPT = 'Follow the instructions in the audio.'


def prompt_for(name):
    if name in ('gsm8k', 'svamp', 'multiarith'):
        return engine.PROMPT
    if name == 'ifeval':
        return IFEVAL_PROMPT
    choices = 'A, B, or C' if name == 'bbh3' else 'A, B, C, D, or E'
    return ('Solve the problem in the audio. End your response with a final answer line in the format:\n'
            'The answer is (<letter>).\nReplace <letter> with '+choices+'.')


def selections(pool, agreed, fit_count, name):
    """Sample from the full pool. No reserved delivery subset or shared3 filtering."""
    order = sorted(pool)
    if len(order) != len(set(order)):
        raise ValueError('Duplicate question IDs.')
    random.Random(42).shuffle(order)
    random_order = sorted(pool)
    random.Random(43).shuffle(random_order)
    result = {}
    for e in engine.EMOTIONS:
        eligible = sorted(set(agreed[e]) & set(pool))
        count = min(fit_count, len(eligible))
        if count < 1 or fit_count >= len(pool):
            raise ValueError(f'{name}/{e}: no recognized clips, or no unseen questions remain.')
        ordinary = set(random_order[:fit_count])
        recognized = set(random.Random(44).sample(eligible, count))
        result[e] = dict(random=[q for q in order if q in ordinary], agreed=[q for q in order if q in recognized])
    return order, result


def scoring_bank(name, bank, records):
    if name not in ('bbh3', 'bbh5', 'ifeval'):
        return bank
    fields = ('key', 'prompt', 'instruction_id_list', 'kwargs') if name == 'ifeval' else ('original_input', 'original_target')
    bank = {q: dict(item) for q, item in bank.items()}
    parents = {Path(r['audio_file']).parent for r in records.values()}
    for parent in sorted(parents):
        for row in read_rows(parent/'questions.jsonl'):
            q = row.get('question_id')
            if q not in bank or not all(k in row for k in fields):
                continue
            if row.get('question', row.get('prompt')) != bank[q]['question']:
                raise ValueError(f'{name}/{q}: scoring question differs from the audio source.')
            if name.startswith('bbh') and engine.task.normal_answer(row['original_target']).strip('()') != bank[q]['answer'].strip('()'):
                raise ValueError(f'{name}/{q}: official option differs from the saved answer.')
            bank[q].update({k: row[k] for k in fields})
    if not all(all(k in item for k in fields) for item in bank.values()):
        from diffa_collection_report import collection_bank
        plan = dict(name=name, bank=bank, collection_ids=list(bank), target={},
                    pairs=[dict(question_id=q, neutral=records[q, 'neutral'], emotion=records[q, 'angry']) for q in bank])
        complete = collection_bank(plan)
        for q in bank:
            bank[q].update({k: complete[q][k] for k in fields})
    return bank


def prepare(name, registry, output, spec=None):
    pool_size, fit_count = spec or SPECS[name]
    bank, records, predictions = {}, {}, {}
    for q in registry.neutral_ids(name):
        neutral = registry.choose(name, q, 'neutral')
        entries = {s: registry.choose(name, q, s, question=neutral['question'], answer=neutral['answer']) for s in STYLES}
        if not all(entries.values()):
            continue
        for s, item in entries.items():
            variants = {x['record']['audio_sha256'] for x in registry.options[name, q, s]
                        if x['question'] == neutral['question'] and x['answer'] == neutral['answer']}
            if len(variants) != 1:
                raise ValueError(f'Ambiguous {name}/{q}/{s} audio; restore the same Jenny bundle.')
            records[q, s] = item['record']
            if s != 'neutral':
                label = registry.labels.get(item['record']['audio_sha256'], {})
                if label.get('classification_status') != 'classified':
                    raise ValueError(f'Missing saved emotion2vec label: {name}/{q}/{s}. No classification is launched automatically.')
                predictions[q, s] = label['emotion']
        bank[q] = dict(question_id=q, question=neutral['question'], answer=neutral['answer'])
    if len(bank) != pool_size:
        raise ValueError(f'{name}: expected all{pool_size} complete-four audio questions, found{len(bank)}.')
    bank = scoring_bank(name, bank, records)
    agreed = {e: [q for q in bank if predictions[q, e] == e] for e in engine.EMOTIONS}
    order, selected = selections(list(bank), agreed, fit_count, name)
    prompt = prompt_for(name)
    recipe = dict(engine.RECIPE, max_new_tokens=1024 if name == 'ifeval' else 512)
    manifest = dict(protocol=PROTOCOL, name=name, prompt=prompt, recipe=recipe, bank=bank,
        delivery_ids=order, fitting_order=order, collections=selected,
        audio_sha256={q: {s: records[q, s]['audio_sha256'] for s in STYLES} for q in sorted(bank)},
        emotion2vec_predictions={q: {e: predictions[q, e] for e in engine.EMOTIONS} for q in sorted(bank)},
        seeds=dict(question_order=42, random_collection=43, agreed_collection=44),
        counts=dict(pool=pool_size, fitting_per_emotion={e: {r: len(selected[e][r]) for r in engine.REGIMES} for e in engine.EMOTIONS}),
        selection='All complete-four questions receive natural generation. Random fitting samples do not filter labels; '
                  'Random uses half the pool; agreed uses min(recognized, half) independently per emotion from the whole pool. Neutral recognition ignored. '
                  'No shared3 or response-accuracy filtering. No reserved delivery subset. Each vector steers all questions outside its own fitting IDs.')
    output.mkdir(parents=True, exist_ok=True)
    engine.task.winners.freeze(output/'split.json', manifest)
    checksum = engine.fingerprint(manifest)
    primary = 'strict_prompt' if name == 'ifeval' else 'flexible-extract' if name.startswith('bbh') else 'flexible-numeric'
    plans = {}
    for e in engine.EMOTIONS:
        plans[e] = {}
        for regime in engine.REGIMES:
            train = selected[e][regime]
            target = dict(name=name, task=name, prompt=prompt, **recipe, bank=bank, primary_metric=primary,
                          records=[records[q, 'neutral'] for q in order], extraction_ids=train)
            plan = dict(protocol=PROTOCOL, name=name, emotion=e, regime=regime, prompt=prompt,
                collection_prompt=prompt, **recipe, split_sha256=checksum, bank=bank, target=target,
                collection_ids=train, test_ids=[], delivery_ids=order,
                pairs=[dict(question_id=q, neutral=records[q, 'neutral'], emotion=records[q, e]) for q in train],
                recordings=[dict(records[q, s], condition=s) for q in order for s in ('neutral', e)],
                averaging='Matched forwards within each pair; token-position means; equal per-question weight.')
            folder = output/e/regime
            folder.mkdir(parents=True, exist_ok=True)
            engine.task.winners.freeze(folder/'run_config.json', plan)
            plans[e][regime] = plan
    print(f'{name}: all{pool_size} natural deliveries; split SHA256={checksum}', flush=True)
    print('Prompt:', repr(prompt), 'Budget:', recipe['max_new_tokens'], flush=True)
    for e in engine.EMOTIONS:
        print(f'{name}/{e}: recognized={len(agreed[e])}; random/agreed fitting={len(selected[e]["random"])}/{len(selected[e]["agreed"])}; '
              f'random/agreed steering questions={pool_size-len(selected[e]["random"])}/{pool_size-len(selected[e]["agreed"])}.', flush=True)
    return manifest, plans


def folder_for(args, name):
    return args.gsm_output if name == 'gsm8k' else args.output/name


def summarize(args):
    for name in args.datasets:
        folder = folder_for(args, name)
        if not (folder/'split.json').exists():
            print(f'{name}: not prepared yet.', flush=True)
            continue
        plans = {e: {r: json.loads((folder/e/r/'run_config.json').read_text()) for r in engine.REGIMES} for e in engine.EMOTIONS}
        print(f'\n{name}', flush=True)
        if args.agreed_only:
            engine.report_agreed_accuracy(plans, folder, args.emotions, agreement_emotions=args.agreement_emotions)
        else:
            engine.report(plans, folder, args.emotions, detailed=args.detailed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'prepare', 'summary'), default='run')
    parser.add_argument('--datasets', choices=DATASETS, nargs='+', default=list(NEXT))
    parser.add_argument('--emotions', choices=engine.EMOTIONS, nargs='+', default=list(engine.EMOTIONS))
    parser.add_argument('--audio-root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('outputs/diffa-vanilla'))
    parser.add_argument('--gsm-output', type=Path, default=Path('outputs/diffa-gsm-vanilla'))
    parser.add_argument('--detailed', action='store_true')
    parser.add_argument('--agreed-only', action='store_true', help='Summary only: accuracy on common e2v-recognized questions.')
    parser.add_argument('--agreement-emotions', nargs='+', choices=engine.EMOTIONS)
    args = parser.parse_args()
    args.datasets = list(dict.fromkeys(args.datasets))
    args.emotions = list(dict.fromkeys(args.emotions))
    if (args.agreed_only or args.agreement_emotions) and args.mode != 'summary':
        parser.error('Agreement filtering is a CPU-only summary option; use --mode summary.')
    if args.agreement_emotions and not args.agreed_only:
        parser.error('--agreement-emotions requires --agreed-only.')
    if args.mode == 'summary':
        summarize(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        own = stack.enter_context((args.output/'runner.lock').open('a'))
        print('Waiting for any previous all-dataset queue; no GPU model loaded.', flush=True)
        fcntl.flock(own, fcntl.LOCK_EX)
        print('CPU preflight: exact Jenny recordings and saved emotion2vec labels. No TTS/classification.', flush=True)
        registry = Registry(args.audio_root)
        cases = []
        # Match the standalone runner lock order: per-run locks BEFORE the shared GPU lock.
        for name in args.datasets:
            folder = folder_for(args, name)
            folder.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((folder/'runner.lock').open('a'))
            print(f'Waiting for existing {name} runner, if any.', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            manifest, plans = (engine.prepare(registry, folder) if name == 'gsm8k'
                               else prepare(name, registry, folder))
            scoped = engine.prepare_deliveries(registry, manifest, plans, folder)
            scoring = folder/'delivery/scoring-preflight'
            scoring.mkdir(parents=True, exist_ok=True)
            engine.task.winners.score_rows(dict(target=scoped['angry']['random']['target']), [], scoring)
            cases.append((name, folder, manifest, plans, scoped))
        if args.mode == 'prepare':
            return
        if shutil.disk_usage(args.output).free < 1024**3:
            raise RuntimeError('Need at least1 GiB free for aggregate checkpoints and reports.')
        Path('outputs').mkdir(exist_ok=True)
        gpu = stack.enter_context(Path('outputs/jenny-task-vectors.lock').open('a'))
        print('Waiting for existing DIFFA GPU work to finish; no model loaded while waiting.', flush=True)
        fcntl.flock(gpu, fcntl.LOCK_EX)
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            time.sleep(30)
        from extract_vectors import load_model
        from diffa_model_receipts import model_revisions
        model, tokenizer, features = load_model()
        for name, folder, manifest, plans, scoped in cases:
            engine.task.winners.atomic_text(folder/'runtime.json', json.dumps(dict(
                runner_sha256=engine.task.winners.digest(__file__), engine_sha256=engine.task.winners.digest(engine.__file__),
                split_sha256=engine.fingerprint(manifest), model_revisions=model_revisions(),
                generator_sha256=engine.task.winners.digest(Path(__file__).with_name('extract_vectors.py')),
                model_config=model.config.to_dict()), indent=2, default=str)+'\n')
            order = manifest.get('delivery_ids', manifest.get('test_ids', [])+manifest['fitting_order'])
            engine.generate_and_collect(plans, scoped, args.emotions, order, folder, model, tokenizer, features)
            engine.report(plans, folder, args.emotions, tokenizer, detailed=args.detailed)
            engine.steer(plans, scoped, args.emotions, folder, model, tokenizer, features)
            print(f'{name}: complete. Natural generation, both vector collections and both x+1 steering runs.', flush=True)


if __name__ == '__main__':
    main()
