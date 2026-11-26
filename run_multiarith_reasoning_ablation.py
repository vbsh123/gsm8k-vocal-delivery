"""Compare two written prompts on the original 290 neutral/angry MultiArith audio pairs."""
import argparse
from importlib.metadata import version
import json
import re
from pathlib import Path
from statistics import mean, median

import diffa_fewshot_emotions as evaluation
from diffa_model_receipts import model_revisions
from multiarith_instruction_checks import THIRD_ASK, checks
from phi4_queued_workers import gpu_queue
from phi4_sweep import job_lock
from rescore_audio_steering import paired
from run_arithmetic_audio_pilot import accuracy_interaction

winners = evaluation.winners
REFERENCE = Path(__file__).parent / 'data/multiarith_original_jenny290.json'
OUTPUT = Path('outputs/diffa-multiarith-original-minus-reasoning290')
SENTENCE = 'Give concise step-by-step reasoning with only the necessary calculations. '
STYLES = ('neutral', 'angry')
THREE_ASKS = Path('outputs/diffa-multiarith-three-asks290')
READY_FORMAT = Path('outputs/diffa-multiarith-ready-format290')
SEQUENCED = Path('outputs/diffa-multiarith-sequenced20')
SEQUENCED_REST = Path('outputs/diffa-multiarith-sequenced-remaining270')
COMPARISON = REFERENCE.with_name('multiarith_three_ask_comparison290.json')
SEQUENCED_PROMPT = (
    'Please solve the math question in the audio. Start your response with the word Ready. '
    'Then give your reasoning for the question. Finally, end your response with a line in this format:\n'
    '#### <numeric answer>\nReplace <numeric answer> with the answer to the question.'
)
EXTRA_STYLES = ('happy', 'sad')


def prepare(folder, three_asks=False, ready_format=False, sequenced=False, remaining=False):
    reference = json.loads(REFERENCE.read_text())
    original = reference['original_prompt']
    if original.count(SENTENCE) != 1 or reference['recipe'] != evaluation.night.vanilla.RECIPE:
        raise ValueError('Unexpected archived MultiArith prompt or decoding settings.')
    questions = reference['questions']
    if len(questions) != 290:
        raise ValueError('Expected all 290 original held-out question IDs.')
    if remaining:
        questions = dict(list(questions.items())[20:])
    elif sequenced:
        questions = dict(list(questions.items())[:20])
    records = []
    for q, item in questions.items():
        for style in STYLES:
            clip = item[style]
            records.append(dict(question_id=q, style=style, condition=style, voice='en-US-JennyNeural',
                                audio_file=clip['audio_file'], audio_sha256=clip['audio_sha256'],
                                reference_answer=item['answer']))
    plan = dict(task='multiarith', name='multiarith', model='diffa', seed=42, max_new_tokens=512,
                recipe=reference['recipe'], bank={q:dict(answer=i['answer']) for q,i in questions.items()},
                records=records, question_order=list(questions), original_prompt=original,
                prompt=original.replace(SENTENCE, '', 1), reference_sha256=winners.digest(REFERENCE),
                primary_metric='flexible-numeric')
    if three_asks:
        plan['prompt'] = original.replace(SENTENCE, SENTENCE + THIRD_ASK + ' ', 1)
        plan['primary_metric'] = 'all-three-observed'
        plan['instruction_checks_sha256'] = winners.digest(Path(__file__).with_name('multiarith_instruction_checks.py'))
        extra = json.loads(REFERENCE.with_name('multiarith_original_instruction_checks.json').read_text())
        if extra['scores_sha256'] != reference['scores_sha256'] or extra['checks_sha256'] != plan['instruction_checks_sha256']:
            raise ValueError('Archived two-ask checks differ from the score source or check definitions.')
        for q, item in questions.items():
            for style in STYLES:
                item[style]['scores'].update(extra['questions'][q][style])
    if ready_format:
        plan.update(prompt=original.replace(SENTENCE, THIRD_ASK + ' ', 1),
                    primary_metric='both-asks-observed',
                    instruction_checks_sha256=winners.digest(Path(__file__).with_name('multiarith_instruction_checks.py')))
    if sequenced or remaining:
        plan.update(prompt=SEQUENCED_PROMPT, variant='sequenced', primary_metric='all-three-observed',
                    instruction_checks_sha256=winners.digest(Path(__file__).with_name('multiarith_instruction_checks.py')))
    if remaining:
        plan.update(primary_metric='exact-final-line', comparison_sha256=winners.digest(COMPARISON))
    config = folder/'run_config.json'
    if config.exists():
        if json.loads(config.read_text()) != plan:
            raise ValueError(f'Archived reference or experiment settings differ from {config}.')
    else:
        winners.freeze(config, plan)
    return plan, questions


def pending_recordings(folder, plan, audio_root):
    done = evaluation.raw_rows(folder, plan, repair=True)
    registry = None
    pending = []
    for record in plan['records']:
        if (record['question_id'], record['condition']) in done:
            continue
        path = Path(record['audio_file'])
        if not path.is_file() or winners.digest(path) != record['audio_sha256']:
            registry = registry or evaluation.night.Registry(evaluation.night.audio_root(audio_root))
            record = registry.relocate(record)
        pending.append(record)
    return pending


def compare_remaining(plan, ids, by_style):
    import numpy as np

    baseline = json.loads(COMPARISON.read_text())
    if baseline['reference_sha256'] != plan['reference_sha256']:
        raise ValueError('Three-ask baseline uses different recording pairs.')
    results = []
    for i, metric in enumerate(baseline['metrics']):
        old = np.array([baseline['questions'][q][s][i] for q in ids for s in STYLES]).reshape(-1, 2)
        new = np.array([[int(by_style[s][q]['metrics'][metric]) for s in STYLES] for q in ids])
        differences = (new[:,1] - new[:,0]) - (old[:,1] - old[:,0])
        # Resample questions, keeping both prompts and both deliveries together.
        values, counts = np.unique(differences, return_counts=True)
        rng = np.random.default_rng(42)
        draws = rng.multinomial(len(ids), counts / len(ids), size=20000)
        ci = np.percentile(100 * (draws @ values) / len(ids), [2.5, 97.5]).tolist()
        gain = float(100 * differences.mean())
        old_counts, new_counts = old.sum(axis=0).tolist(), new.sum(axis=0).tolist()
        results.append(dict(metric=metric, pairs=len(ids), original=old_counts, sequenced=new_counts,
                            gap_change_pp=gain, bootstrap_ci95_pp=ci))
        print(f'{metric} ({len(ids)}/270 remaining pairs): original neutral/angry {old_counts}, '
              f'sequenced {new_counts}, gap change {gain:+.2f} pp, paired bootstrap 95% CI {ci}')
    return results


def report(folder, plan, questions):
    saved = evaluation.raw_rows(folder, plan)
    ids = [q for q in plan['question_order'] if all((q, s) in saved for s in STYLES)]
    print(f'MultiArith: {len(saved)}/{len(plan["records"])} new responses, {len(ids)}/{len(questions)} complete neutral/angry pairs.')
    if not ids:
        return
    rows = [saved[q,s] for q in ids for s in STYLES]
    scored, receipt = winners.score_rows(dict(target=plan), rows, folder/'scoring')
    sequenced = plan.get('variant') == 'sequenced'
    if sequenced or plan['primary_metric'] == 'both-asks-observed':
        by_style = {style: {} for style in STYLES}
        for row in scored:
            measured = checks(row['response'])
            opening, final = measured['ready-opening'], measured['exact-final-line']
            row['metrics'].update({'ready-opening': opening, 'exact-final-line': final,
                                   'both-asks-observed': opening and final})
            if sequenced:
                row['metrics'].update(measured)
                row['metrics']['ready-only'] = bool(re.fullmatch(r'\s*Ready[.!?]?\s*', row['response'], re.I))
                row['metrics']['number-extracted'] = row['predictions']['flexible-extract'] != '[invalid]'
            by_style[row['condition']][row['question_id']] = row
        lengths = {}
        for style, group in by_style.items():
            tokens = [r['response_tokens'] for r in group.values()]
            lengths[style] = dict(mean=mean(tokens), median=median(tokens))
            print(f'{style}: mean/median tokens {mean(tokens):.1f}/{median(tokens):g}')
        comparisons = []
        metrics = ('ready-opening', 'exact-final-line', 'both-asks-observed', 'flexible-numeric')
        if sequenced:
            metrics = ('ready-only', 'number-extracted', 'ready-opening', 'written-operation',
                       'exact-final-line', 'all-three-observed', 'flexible-numeric')
        for metric in metrics:
            result = paired([(by_style['neutral'][q]['metrics'][metric], by_style['angry'][q]['metrics'][metric])
                             for q in ids])
            comparisons.append(dict(metric=metric, **result))
            print(f'{metric}: neutral {result["baseline_correct"]}/{len(ids)} -> '
                  f'angry {result["steered_correct"]}/{len(ids)}, {result["gain_pp"]:+.2f} pp, '
                  f'95% CI {result["ci95_gain_pp"]}, p={result["mcnemar_exact_p"]:.4g}')
        summary = dict(matched=len(ids), comparisons=comparisons, token_lengths=lengths, scorer=receipt)
        if 'comparison_sha256' in plan:
            summary['remaining_prompt_comparison'] = compare_remaining(plan, ids, by_style)
        winners.atomic_text(folder/'summary.json', json.dumps(summary, indent=2)+'\n')
        winners.atomic_text(folder/'rescored.jsonl', ''.join(json.dumps(r)+'\n' for r in scored))
        return
    three_asks = 'instruction_checks_sha256' in plan
    label_new = 'three asks' if three_asks else 'without reasoning sentence'
    label_original = 'original (two asks)' if three_asks else 'original'
    change_label = 'after adding Ready' if three_asks else 'after removal'
    if three_asks:
        for row in scored:
            row['metrics'].update(checks(row['response']))
    groups = {s:{q:dict(metrics=questions[q][s]['scores']) for q in ids} for s in STYLES}
    lengths = {}
    for s in STYLES:
        group = {r['question_id']:r for r in scored if r['condition'] == s}
        # accuracy_interaction subtracts its unsuffixed groups from its suffixed groups.
        groups[s+'-reasoning'] = group
        tokens = [r['response_tokens'] for r in group.values()]
        lengths[s] = dict(mean=mean(tokens), median=median(tokens))
        print(f'{label_new} / {s}: mean/median tokens {mean(tokens):.1f}/{median(tokens):g}')
    comparisons, changes = [], []
    for metric in scored[0]['metrics']:
        for label, suffix in [(label_original, ''), (label_new, '-reasoning')]:
            a, b = groups['neutral'+suffix], groups['angry'+suffix]
            result = paired([(a[q]['metrics'][metric], b[q]['metrics'][metric]) for q in ids])
            comparisons.append(dict(prompt=label, metric=metric, **result))
            print(f'{label} / {metric}: neutral {result["baseline_correct"]}/{len(ids)} -> '
                  f'angry {result["steered_correct"]}/{len(ids)}, {result["gain_pp"]:+.2f} pp, '
                  f'95% CI {result["ci95_gain_pp"]}, p={result["mcnemar_exact_p"]:.4g}')
        change = accuracy_interaction(groups, metric)
        changes.append(change)
        print(f'{metric} / change in angry-neutral gap {change_label}: {change["interaction_pp"]:+.2f} pp, '
              f'95% CI {change["ci95_pp"]}, p={change["p"]}')
    winners.atomic_text(folder/'summary.json', json.dumps(dict(matched=len(ids), comparisons=comparisons,
                        changes=changes, new_token_lengths=lengths, scorer=receipt), indent=2)+'\n')
    winners.atomic_text(folder/'rescored.jsonl', ''.join(json.dumps(r)+'\n' for r in scored))


def emotion_plan(folder, baseline):
    if not (baseline/'run_config.json').is_file():
        raise FileNotFoundError(f'Expected the completed Jenny three-ask run in {baseline}.')
    original, _ = prepare(baseline, three_asks=True)
    clips = json.loads(REFERENCE.with_name('multiarith_three_asks_audio.json').read_text())
    records = [dict(question_id=q, style=s, condition=s, voice='en-US-JennyNeural',
                    reference_answer=original['bank'][q]['answer'], **clips[q][s])
               for q in original['question_order'] for s in EXTRA_STYLES]
    plan = dict(original, records=records, baseline_sha256=winners.digest(baseline/'run_config.json'))
    winners.freeze(folder/'run_config.json', plan)
    return plan, original


def emotion_report(folder, plan, baseline, original):
    saved = evaluation.raw_rows(baseline, original)
    new = evaluation.raw_rows(folder, plan)
    saved.update(new)
    styles = STYLES + EXTRA_STYLES
    ids = [q for q in plan['question_order'] if all((q, s) in saved for s in styles)]
    print(f'Jenny MultiArith: {len(new)}/580 happy/sad responses, {len(ids)}/290 four-way matched questions.')
    if not ids:
        return
    scored, receipt = winners.score_rows(dict(target=plan), [saved[q,s] for q in ids for s in styles], folder/'scoring')
    groups = {s: {} for s in styles}
    for row in scored:
        row['metrics'].update(checks(row['response']))
        groups[row['condition']][row['question_id']] = row
    lengths, comparisons = {}, []
    for s in styles:
        tokens = [r['response_tokens'] for r in groups[s].values()]
        lengths[s] = dict(mean=mean(tokens), median=median(tokens))
        print(f'{s}: mean/median tokens {mean(tokens):.1f}/{median(tokens):g}')
    for metric in scored[0]['metrics']:
        for s in styles[1:]:
            result = paired([(groups['neutral'][q]['metrics'][metric], groups[s][q]['metrics'][metric]) for q in ids])
            comparisons.append(dict(emotion=s, metric=metric, **result))
            print(f'{metric}: neutral {result["baseline_correct"]}/{len(ids)} -> {s} '
                  f'{result["steered_correct"]}/{len(ids)}, {result["gain_pp"]:+.2f} pp, '
                  f'95% CI {result["ci95_gain_pp"]}, p={result["mcnemar_exact_p"]:.4g}')
    # Cochran's Q compares three paired binary outcomes on the same questions.
    from scipy.stats import chi2
    metric = plan['primary_metric']
    bits = [[int(groups[s][q]['metrics'][metric]) for s in styles[1:]] for q in ids]
    totals = list(map(sum, zip(*bits)))
    k = len(totals)
    denom = k * sum(totals) - sum(sum(row)**2 for row in bits)
    statistic = (k-1) * (k*sum(v*v for v in totals)-sum(totals)**2) / denom if denom else 0.
    omnibus = dict(metric=metric, styles=styles[1:], q=statistic, df=k-1,
                   discordant_questions=sum(0 < sum(row) < k for row in bits),
                   p_asymptotic=float(chi2.sf(statistic, k-1)) if denom else 1.)
    print(f'{metric} across angry/happy/sad: paired Cochran Q={statistic:.2f}, '
          f'asymptotic p={omnibus["p_asymptotic"]:.4g}, discordant questions={omnibus["discordant_questions"]}/{len(ids)}')
    winners.atomic_text(folder/'summary.json', json.dumps(dict(matched=len(ids), comparisons=comparisons,
                        token_lengths=lengths, emotion_test=omnibus, scorer=receipt), indent=2)+'\n')
    winners.atomic_text(folder/'rescored.jsonl', ''.join(json.dumps(r)+'\n' for r in scored))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'summary', 'prepare'), default='run')
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument('--sequenced-rest', action='store_true', help='Continue the sequenced pilot on questions 21–290.')
    variants.add_argument('--sequenced', action='store_true', help='Test explicit ordering on the first 20 matched pairs.')
    variants.add_argument('--ready-format', action='store_true', help='Ready opening and exact final line only.')
    variants.add_argument('--three-asks', action='store_true', help='Add the Ready opening to the original reasoning and formatting instructions.')
    parser.add_argument('--happy-sad', action='store_true', help='Extend the completed Jenny three-ask run.')
    parser.add_argument('--baseline', type=Path, default=THREE_ASKS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--audio-root', type=Path)
    args = parser.parse_args()
    folder = args.output or (THREE_ASKS if args.three_asks else OUTPUT)
    if (args.sequenced or args.sequenced_rest) and args.happy_sad:
        parser.error('--sequenced uses neutral and angry only.')
    if args.sequenced:
        folder = args.output or SEQUENCED
    if args.sequenced_rest:
        folder = args.output or SEQUENCED_REST
    if args.ready_format:
        if args.happy_sad:
            parser.error('--ready-format uses neutral and angry only.')
        folder = args.output or READY_FORMAT
    if args.happy_sad:
        folder = args.output or THREE_ASKS.with_name(THREE_ASKS.name+'-happy-sad')
        if folder.resolve() == args.baseline.resolve():
            parser.error('Use a separate output for the new recordings.')
    if args.happy_sad and args.mode == 'summary':
        if (folder/'run_config.json').exists():
            plan = json.loads((folder/'run_config.json').read_text())
            original, _ = prepare(args.baseline, three_asks=True)
            if plan['baseline_sha256'] != winners.digest(args.baseline/'run_config.json'):
                raise ValueError('Three-ask baseline changed.')
            emotion_report(folder, plan, args.baseline, original)
        else:
            print('Jenny MultiArith: 0/580 happy/sad responses.')
        return
    if args.mode == 'summary':
        if (folder/'run_config.json').is_file():
            plan, questions = prepare(folder, args.three_asks, args.ready_format, args.sequenced, args.sequenced_rest)
            report(folder, plan, questions)
        else:
            size = 270 if args.sequenced_rest else (20 if args.sequenced else 290)
            print(f'MultiArith: 0/{2*size} responses, 0/{size} pairs.')
        return
    with job_lock(folder):
        if args.happy_sad:
            plan, original = emotion_plan(folder, args.baseline)
            if len(evaluation.raw_rows(args.baseline, original)) != len(original['records']):
                raise ValueError('The original neutral/angry run is incomplete.')
        else:
            plan, questions = prepare(folder, args.three_asks, args.ready_format, args.sequenced, args.sequenced_rest)
        print(plan['prompt'], flush=True)
        pending = pending_recordings(folder, plan, args.audio_root)
        print(f'{len(pending)} verified recordings pending.', flush=True)
        if args.mode == 'prepare':
            return
        if pending:
            winners.score_rows(dict(target=plan), [], folder/'scoring')
            with gpu_queue():
                from src.dataloader import build_chat_prompt
                backend = evaluation.night.DiffaBackend()
                text, count = evaluation.context_guard(backend.tokenizer, plan['prompt'], build_chat_prompt)
                winners.atomic_text(folder/'native-chat-prompt.txt', text)
                winners.freeze(folder/'runtime.json', dict(model_revisions=model_revisions(),
                               packages={k:version(k) for k in ('torch','transformers')}, native_prompt_tokens=count))
                evaluation.generate_dataset(folder, plan, pending, backend)
        if args.happy_sad:
            emotion_report(folder, plan, args.baseline, original)
        else:
            report(folder, plan, questions)


if __name__ == '__main__':
    main()
