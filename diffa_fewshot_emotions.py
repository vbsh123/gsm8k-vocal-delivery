"""Published CoT demonstrations + audio targets, all four natural emotions; evaluation only.

This is an audio adaptation, not an official text-benchmark replication. Native
DIFFA chat/audio insertion and 512-token decoding are retained. No activations,
vectors, E2V filtering, response rewriting or historical response reuse.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import re
from statistics import mean, median
import subprocess
import time
from urllib.request import Request, urlopen

import emotion_block_night as night
from rescore_audio_steering import HarnessTaskScorer, numeric_equal, paired

winners = night.winners
PROTOCOL = 'diffa-published-fewshot-audio-eval-v1'
TOTALS = dict(bbh3=250, bbh5=250, gsm8k=1319)
SOURCE_MANIFEST = Path(__file__).resolve().parent/'data/fewshot_prompt_sources.json'
BBH_SUFFIX = "Q: The question and answer choices are provided in the audio.\nA: Let's think step by step."
GSM_SUFFIX = 'Q: Solve the math problem in the audio.\nA:'


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def prompt_source(name, folder):
    source = json.loads(SOURCE_MANIFEST.read_text())[name]
    path = folder/'sources'/Path(source['url']).name
    if path.exists():
        data = path.read_bytes()
    else:
        print(f'Downloading pinned public prompt source: {source["url"]}', flush=True)
        with urlopen(Request(source['url'], headers={'User-Agent': PROTOCOL}), timeout=120) as response:
            data = response.read()
        if digest_bytes(data) != source['sha256']:
            raise ValueError('Downloaded prompt source SHA256 differs; nothing will run.')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if digest_bytes(data) != source['sha256']:
        raise ValueError(f'Cached prompt source changed: {path}')
    return data.decode('utf-8'), source


def render_prompt(name, text):
    if name in ('bbh3', 'bbh5'):
        # Authors put a training-contamination canary before the actual prompt.
        header, separator, body = text.partition('\n-----\n')
        if not separator or not header.startswith('BENCHMARK DATA SHOULD NEVER'):
            raise ValueError('Unexpected BBH author prompt header.')
        questions = re.findall(r'^Q: (.*?)\nA:', body, re.M | re.S)
        if len(questions) != 3 or body.count("A: Let's think step by step.") != 3:
            raise ValueError('Expected three unmodified BBH worked examples.')
        return body.rstrip()+'\n\n'+BBH_SUFFIX, questions, BBH_SUFFIX
    if name != 'gsm8k':
        raise ValueError('Unsupported few-shot dataset.')
    import yaml
    config = yaml.safe_load(text)
    samples = config['fewshot_config']['samples']
    if config['task'] != 'gsm8k_cot' or config['num_fewshot'] != 8 or len(samples) != 8:
        raise ValueError('Expected the pinned eight-shot GSM8K CoT task.')
    examples = ['Q: '+item['question']+'\nA: '+item['target'] for item in samples]
    return '\n\n'.join(examples)+'\n\n'+GSM_SUFFIX, [s['question'] for s in samples], GSM_SUFFIX


def no_demo_overlap(questions, bank):
    normalize = lambda text: ' '.join(text.split()).casefold()
    demos = {normalize(q) for q in questions}
    overlap = [q for q, item in bank.items()
               if normalize(item.get('original_input', item['question'])) in demos
               or normalize(item['question']) in demos]
    if overlap:
        raise ValueError(f'Published demonstrations overlap evaluation questions: {overlap}')


def full_cohort(name, registry):
    """All questions, without requiring labels or creating a fitting split."""
    bank, records = {}, {}
    for q in registry.neutral_ids(name):
        neutral = registry.choose(name, q, 'neutral')
        for style in night.STYLES:
            item = registry.choose(name, q, style, question=neutral['question'], answer=neutral['answer'])
            if item is None:
                raise ValueError(f'Missing {name}/{q}/{style}; restore existing Jenny audio.')
            variants = {x['record']['audio_sha256'] for x in registry.options[name, q, style]
                        if x['question'] == neutral['question'] and x['answer'] == neutral['answer']}
            if len(variants) != 1:
                raise ValueError(f'Ambiguous {name}/{q}/{style} recordings.')
            records[q, style] = dict(item['record'], condition=style)
        bank[q] = dict(question_id=q, question=neutral['question'], answer=neutral['answer'])
    if len(bank) != TOTALS[name]:
        raise ValueError(f'{name}: need all {TOTALS[name]} questions; found {len(bank)}.')
    bank = night.catalog.scoring_bank(name, bank, records)
    order = sorted(bank)
    random.Random(42).shuffle(order)
    return bank, order, [records[q, s] for q in order for s in night.STYLES]


def validate_plan(plan):
    name = plan['name']
    ids = plan['question_order']
    records = plan['records']
    expected = {(q, s) for q in ids for s in night.STYLES}
    if (plan['protocol'] != PROTOCOL or plan['model'] != 'diffa'
            or len(ids) != TOTALS[name] or len(set(ids)) != len(ids)
            or set(ids) != set(plan['bank']) or len(records) != len(expected)
            or {(r['question_id'], r['condition']) for r in records} != expected
            or plan['recipe'] != night.vanilla.RECIPE or plan['max_new_tokens'] != 512):
        raise ValueError('Few-shot run must retain every question/style and the native DIFFA recipe.')
    for record in records:
        if (record['voice'] != 'en-US-JennyNeural' or record['style'] != record['condition']
                or record['reference_answer'] != plan['bank'][record['question_id']]['answer']):
            raise ValueError('Audio identity or saved gold reference differs.')


def prepare(name, source, output, registry=None):
    folder = output/name
    folder.mkdir(parents=True, exist_ok=True)
    raw_source, reference = prompt_source(name, folder)
    prompt, demos, suffix = render_prompt(name, raw_source)
    existing = folder/'run_config.json'
    if existing.exists():
        # Re-running never silently rebuilds a cohort from a different machine.
        plan = json.loads(existing.read_text())
    else:
        parent = source/'diffa'/name/'run_config.json'
        if parent.exists():
            old = json.loads(parent.read_text())
            manifest = json.loads((parent.parent/'cohort.json').read_text())
            if (old['model'] != 'diffa' or old['name'] != name
                    or night.vanilla.fingerprint(manifest) != old['cohort_sha256']):
                raise ValueError('Parent natural cohort receipt differs.')
            bank, order, records = deepcopy(old['bank']), old['question_order'], deepcopy(old['records'])
            recipe = old['recipe']
            parent_info = dict(config_sha256=winners.digest(parent), cohort_sha256=old['cohort_sha256'])
        else:
            if registry is None:
                raise FileNotFoundError(f'{parent}: supply the extracted Jenny audio for this dataset.')
            bank, order, records = full_cohort(name, registry)
            recipe = dict(night.vanilla.RECIPE)
            parent_info = None
        plan = dict(protocol=PROTOCOL, name=name, model='diffa', task=name, bank=bank,
                    question_order=order, records=records, recipe=recipe, seed=42, max_new_tokens=512,
                    prompt=prompt, source=reference, audio_target_suffix=suffix,
                    demonstration_count=len(demos), parent=parent_info,
                    scope='All natural deliveries only; no fitting, vectors or steering.',
                    adaptation='Published text demonstrations; target question/choices supplied only as Jenny audio. '
                               'Native DIFFA chat wrapping/audio insertion, 512-token budget and decoding retained. '
                               'No harness generation stop strings or retrospective response truncation.')
    if (plan['prompt'] != prompt or plan['source'] != reference
            or plan['audio_target_suffix'] != suffix or plan['demonstration_count'] != len(demos)):
        raise ValueError('Frozen few-shot prompt differs; use a separate output directory.')
    validate_plan(plan)
    no_demo_overlap(demos, plan['bank'])
    winners.freeze(existing, plan)
    winners.atomic_text(folder/'prompt.txt', prompt+'\n')
    winners.freeze(folder/'prompt-provenance.json', dict(
        source=reference, prompt_sha256=digest_bytes(prompt.encode()), audio_target_suffix=suffix,
        demonstration_count=len(demos), demonstration_questions=demos,
        bbh5_note='Published BBH5 file uses the same three-object demonstrations as BBH3.' if name == 'bbh5' else None,
        adaptation=plan['adaptation']))
    print(f'{name}: {len(plan["question_order"])} questions × all four natural styles; '
          f'{len(demos)} published demonstrations. Full prompt: {folder}/prompt.txt', flush=True)
    print('Audio target suffix:\n'+suffix, flush=True)
    return folder, plan


def scorer_for(name, folder):
    task = 'gsm8k_cot' if name == 'gsm8k' else (
        'bbh_cot_zeroshot_logical_deduction_'+('three' if name == 'bbh3' else 'five')+'_objects')
    scorer = HarnessTaskScorer(task, folder/'scoring')
    if name == 'gsm8k':
        # Bind generation demonstrations AND saved-answer scoring to the same config.
        config_hash = next(v for k, v in scorer.provenance['files'].items() if k.endswith('/gsm8k-cot.yaml'))
        if config_hash != json.loads(SOURCE_MANIFEST.read_text())['gsm8k']['sha256']:
            raise ValueError('Installed GSM8K CoT config differs from pinned source; inspect harness version.')
    return scorer


def raw_rows(folder, plan, repair=False):
    done = night.raw_rows(folder, plan, repair=repair)
    for row in done.values():
        if any(row.get(k) != v for k, v in plan['recipe'].items()):
            raise ValueError('Saved natural response decoding recipe differs.')
    return done


def generate_dataset(folder, plan, runtime_records, backend):
    done = raw_rows(folder, plan, repair=True)
    checksum = winners.digest(folder/'run_config.json')
    with (folder/'deliveries.jsonl').open('a') as stream:
        for record in runtime_records:
            key = record['question_id'], record['condition']
            if key in done:
                continue
            print(f'{plan["name"]} [{len(done)+1}/{len(plan["records"])}] {key[0]}/{key[1]}', flush=True)
            trace = backend.generate(record, plan, capture=False)
            row = dict(record, response=trace['response'], response_tokens=trace['response_tokens'],
                       audio_encoding=trace['audio_encoding'], model='diffa', run_config_sha256=checksum,
                       prompt=plan['prompt'], strength=0., interventions=[], **plan['recipe'])
            stream.write(json.dumps(row)+'\n')
            stream.flush()
            done[key] = row


def context_guard(tokenizer, prompt, builder):
    """Check the exact native text wrapper before its hard-coded 4096 truncation."""
    messages = [dict(role='system', content=''), dict(role='user', content='This is the audio: <audio>. '+prompt)]
    text = builder(messages, add_generation_prompt=True)
    ids = tokenizer(text, truncation=False)['input_ids']
    if len(ids) > 4096:
        raise ValueError(f'Native prompt is {len(ids)} tokens; would silently truncate at 4096. Aborting.')
    return text, len(ids)


def report(folder, plan, scorer=None):
    scorer = scorer or scorer_for(plan['name'], folder)
    rows = list(raw_rows(folder, plan).values())
    bank = plan['bank']
    docs = [dict(question=bank[r['question_id']]['question'], answer=r['reference_answer'])
            if plan['name'] == 'gsm8k' else dict(input=bank[r['question_id']]['original_input'],
                                              target=bank[r['question_id']]['original_target']) for r in rows]
    values = scorer.score([r['response'] for r in rows], docs)
    scored = []
    for i, row in enumerate(rows):
        metrics = {m: bits[i]['correct'] for m, bits in values.items()}
        predictions = {m: bits[i]['prediction'] for m, bits in values.items()}
        if plan['name'] == 'gsm8k':
            metrics['flexible-numeric'] = numeric_equal(predictions['flexible-extract'], row['reference_answer'])
            metrics['format-compliant'] = predictions['strict-match'] != '[invalid]'
        scored.append(dict(row, metrics=metrics, predictions=predictions))
    groups = {s: {r['question_id']: r for r in scored if r['condition'] == s} for s in night.STYLES}
    lengths, accuracy, comparisons = {}, [], []
    print(f'\n{plan["name"]}: published few-shot audio adaptation; {len(rows)}/{len(plan["records"])} natural deliveries.', flush=True)
    for style, group in groups.items():
        if not group:
            continue
        tokens = [r['response_tokens'] for r in group.values()]
        lengths[style] = dict(n=len(tokens), mean_tokens=mean(tokens), median_tokens=median(tokens))
        print(f'{style}: N={len(group)}; mean/median tokens={mean(tokens):.1f}/{median(tokens):g}', flush=True)
        for metric in next(iter(group.values()))['metrics']:
            correct = sum(r['metrics'][metric] for r in group.values())
            accuracy.append(dict(condition=style, metric=metric, n=len(group), correct=correct))
            print(f'  {metric}: {correct}/{len(group)} ({correct/len(group):.1%})', flush=True)
    for style in night.EMOTIONS:
        ids = sorted(groups['neutral'].keys() & groups[style].keys())
        if not ids:
            continue
        matched_lengths = {}
        for s in ('neutral', style):
            tokens = [groups[s][q]['response_tokens'] for q in ids]
            matched_lengths[s] = dict(mean_tokens=mean(tokens), median_tokens=median(tokens))
        for metric in groups[style][ids[0]]['metrics']:
            stats = paired([(groups['neutral'][q]['metrics'][metric], groups[style][q]['metrics'][metric]) for q in ids])
            comparisons.append(dict(reference='neutral', test=style, metric=metric,
                                    matched_lengths=matched_lengths, **stats))
            print(f'{style} vs neutral / {metric}: N={len(ids)}; gain={stats["gain_pp"]:+.2f} pp; '
                  f'p={stats["mcnemar_exact_p"]:.4g}; CI={stats["ci95_gain_pp"]}', flush=True)
    winners.atomic_text(folder/'delivery-rescored.jsonl', ''.join(json.dumps(r)+'\n' for r in scored))
    winners.atomic_text(folder/'delivery-summary.json', json.dumps(dict(lengths=lengths, accuracy=accuracy,
        comparisons=comparisons, scorer=scorer.provenance, protocol=PROTOCOL,
        prompt_sha256=digest_bytes(plan['prompt'].encode()), adaptation=plan['adaptation'],
        statistics='Unadjusted exact paired McNemar p; paired bootstrap 95% CI. Partial pairs explicitly counted.'), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', choices=tuple(TOTALS), nargs='+', default=['bbh3', 'bbh5'])
    parser.add_argument('--mode', choices=('prepare', 'run', 'summary'), default='run')
    parser.add_argument('--source', type=Path, default=Path('outputs/emotion-block-night'))
    parser.add_argument('--output', type=Path, default=Path('outputs/diffa-fewshot-emotions'))
    parser.add_argument('--audio-root', type=Path)
    args = parser.parse_args()
    if args.mode == 'summary':
        for name in args.datasets:
            folder = args.output/name
            if not (folder/'run_config.json').exists():
                print(f'{name}: few-shot run not prepared yet.', flush=True)
                continue
            plan = json.loads((folder/'run_config.json').read_text())
            validate_plan(plan)
            report(folder, plan)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    # Per-dataset locks allow separate queues while preventing duplicate writers.
    from contextlib import ExitStack
    with ExitStack() as stack:
        for name in sorted(set(args.datasets)):
            own = stack.enter_context((args.output/(name+'-runner.lock')).open('a'))
            try:
                fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(f'{name}: another few-shot runner is active. Do not start a duplicate.')
        registry = None
        if any(not (args.output/n/'run_config.json').exists()
               and not (args.source/'diffa'/n/'run_config.json').exists() for n in args.datasets):
            registry = night.Registry(night.audio_root(args.audio_root))
        cases = [prepare(name, args.source, args.output, registry) for name in args.datasets]
        scorers = {p['name']: scorer_for(p['name'], f) for f, p in cases}
        if args.mode == 'prepare':
            print('Prepared prompts/cohorts and scoring preflight only; no model generation.', flush=True)
            return
        pending = [(f, p) for f, p in cases if len(raw_rows(f, p)) < len(p['records'])]
        if pending:
            registry = registry or night.Registry(night.audio_root(args.audio_root))
            relocated = {p['name']: [registry.relocate(r) for r in p['records']] for _, p in pending}
            with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
                print('Waiting for shared GPU lock and idle GPU.', flush=True)
                fcntl.flock(gpu, fcntl.LOCK_EX)
                while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                    time.sleep(30)
                backend = night.DiffaBackend()
                from src.dataloader import build_chat_prompt
                from diffa_model_receipts import model_revisions
                for folder, plan in pending:
                    text, count = context_guard(backend.tokenizer, plan['prompt'], build_chat_prompt)
                    winners.atomic_text(folder/'native-chat-prompt.txt', text)
                    runtime = dict(model_revisions=model_revisions(),
                        packages={k: version(k) for k in ('torch', 'transformers')},
                        model_config=backend.model.config.to_dict(), native_prompt_tokens=count,
                        native_chat_sha256=digest_bytes(text.encode()),
                        runner_sha256=winners.digest(__file__), generator_sha256=winners.digest(night.vanilla.task.__file__))
                    winners.freeze(folder/'runtime.json', runtime)
                    print(f'{plan["name"]}: native prompt {count}/4096 text tokens, no truncation.', flush=True)
                    generate_dataset(folder, plan, relocated[plan['name']], backend)
                    report(folder, plan, scorers[plan['name']])
        for folder, plan in cases:
            if (folder, plan) not in pending:
                report(folder, plan, scorers[plan['name']])


if __name__ == '__main__':
    main()
