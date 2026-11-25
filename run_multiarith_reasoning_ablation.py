"""Compare two written prompts on the original 290 neutral/angry MultiArith audio pairs."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
from statistics import mean, median

import diffa_fewshot_emotions as evaluation
from diffa_model_receipts import model_revisions
from phi4_queued_workers import gpu_queue
from phi4_sweep import job_lock
from rescore_audio_steering import paired
from run_arithmetic_audio_pilot import accuracy_interaction

winners = evaluation.winners
REFERENCE = Path(__file__).parent / 'data/multiarith_original_jenny290.json'
OUTPUT = Path('outputs/diffa-multiarith-original-minus-reasoning290')
SENTENCE = 'Give concise step-by-step reasoning with only the necessary calculations. '
STYLES = ('neutral', 'angry')


def prepare(folder):
    reference = json.loads(REFERENCE.read_text())
    original = reference['original_prompt']
    if original.count(SENTENCE) != 1 or reference['recipe'] != evaluation.night.vanilla.RECIPE:
        raise ValueError('Unexpected archived MultiArith prompt or decoding settings.')
    questions = reference['questions']
    if len(questions) != 290:
        raise ValueError('Expected all 290 original held-out question IDs.')
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


def report(folder, plan, questions):
    saved = evaluation.raw_rows(folder, plan)
    ids = [q for q in plan['question_order'] if all((q, s) in saved for s in STYLES)]
    print(f'MultiArith: {len(saved)}/580 new responses, {len(ids)}/290 complete neutral/angry pairs.')
    if not ids:
        return
    rows = [saved[q,s] for q in ids for s in STYLES]
    scored, receipt = winners.score_rows(dict(target=plan), rows, folder/'scoring')
    groups = {s:{q:dict(metrics=questions[q][s]['scores']) for q in ids} for s in STYLES}
    lengths = {}
    for s in STYLES:
        group = {r['question_id']:r for r in scored if r['condition'] == s}
        # accuracy_interaction subtracts its unsuffixed groups from its suffixed groups.
        groups[s+'-reasoning'] = group
        tokens = [r['response_tokens'] for r in group.values()]
        lengths[s] = dict(mean=mean(tokens), median=median(tokens))
        print(f'Without reasoning sentence / {s}: mean/median tokens {mean(tokens):.1f}/{median(tokens):g}')
    comparisons, changes = [], []
    for metric in scored[0]['metrics']:
        for label, suffix in [('original', ''), ('without reasoning sentence', '-reasoning')]:
            a, b = groups['neutral'+suffix], groups['angry'+suffix]
            result = paired([(a[q]['metrics'][metric], b[q]['metrics'][metric]) for q in ids])
            comparisons.append(dict(prompt=label, metric=metric, **result))
            print(f'{label} / {metric}: neutral {result["baseline_correct"]}/{len(ids)} -> '
                  f'angry {result["steered_correct"]}/{len(ids)}, {result["gain_pp"]:+.2f} pp, '
                  f'95% CI {result["ci95_gain_pp"]}, p={result["mcnemar_exact_p"]:.4g}')
        change = accuracy_interaction(groups, metric)
        changes.append(change)
        print(f'{metric} / change in angry-neutral gap after removal: {change["interaction_pp"]:+.2f} pp, '
              f'95% CI {change["ci95_pp"]}, p={change["p"]}')
    winners.atomic_text(folder/'summary.json', json.dumps(dict(matched=len(ids), comparisons=comparisons,
                        changes=changes, new_token_lengths=lengths, scorer=receipt), indent=2)+'\n')
    winners.atomic_text(folder/'rescored.jsonl', ''.join(json.dumps(r)+'\n' for r in scored))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'summary', 'prepare'), default='run')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--audio-root', type=Path)
    args = parser.parse_args()
    folder = args.output
    if args.mode == 'summary':
        if (folder/'run_config.json').is_file():
            plan, questions = prepare(folder)
            report(folder, plan, questions)
        else:
            print('MultiArith: 0/580 responses, 0/290 pairs.')
        return
    with job_lock(folder):
        plan, questions = prepare(folder)
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
        report(folder, plan, questions)


if __name__ == '__main__':
    main()
