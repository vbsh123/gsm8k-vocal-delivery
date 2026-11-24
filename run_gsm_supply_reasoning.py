"""Compare angry vs neutral on matched GSM8K or MultiArith audio with and without a reasoning request."""
import argparse
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import random
from statistics import mean, median
import subprocess
import time

import diffa_fewshot_emotions as evaluation
from rescore_audio_steering import paired
from run_arithmetic_audio_pilot import accuracy_interaction

night = evaluation.night
winners = night.winners
PROMPT = ('Solve the math problem in the audio. Supply reasoning before the final answer. '
          'End your response with a final answer line in the format:\n#### <number>')
STYLES = ('neutral', 'angry')
PARENT = Path('outputs/emotion-block-night/diffa/gsm8k')
OUTPUT = Path('outputs/diffa-gsm-supply-reasoning200')
PRIMARY = 'flexible-numeric'


DATASETS = {'gsm8k': ('GSM8K', PARENT, OUTPUT),
           'multiarith': ('MultiArith', PARENT.with_name('multiarith'), Path('outputs/diffa-multiarith-supply-reasoning200'))}
# Existing GSM8K receipts were created by the original single-dataset runner.
GSM_RUNNER_V1 = '56ec9ee6a46836af46d015f05aaaed72d36e876c14a6bafe35e586f6d5e28b26'


def prepare(parent, output, dataset='gsm8k'):
    config = parent/'run_config.json'
    original = json.loads(config.read_text())
    if (original['protocol'] != night.PROTOCOL or original['task'] != dataset or original['model'] != 'diffa'
            or original['prompt'] != night.vanilla.PROMPT or original['recipe'] != night.vanilla.RECIPE
            or original['max_new_tokens'] != 512 or original['seed'] != 42 or len(original['bank']) != night.SPECS[dataset][0]):
        raise ValueError(f'Expected the full DIFFA {DATASETS[dataset][0]} bare-task night run.')
    # Choose question IDs before accessing any saved responses.
    ids = random.Random(42).sample(sorted(original['bank']), 200)
    bare = evaluation.raw_rows(parent, original)
    records = {(r['question_id'], r['condition']): r for r in original['records']}
    selected, baselines = [], []
    for q in ids:
        for style in STYLES:
            key = q, style
            if key not in bare:
                raise ValueError(f'Missing saved bare answer for fixed question {q}/{style}.')
            record, row = records[key], bare[key]
            if (record['style'] != style or row['style'] != style or record['voice'] != 'en-US-JennyNeural'
                    or type(row.get('response_tokens')) is not int or row['response_tokens'] < 0):
                raise ValueError(f'Unexpected style, voice or saved token count for {key}.')
            selected.append(record)
            baselines.append(row)
    plan = dict(protocol=f'diffa-{dataset}-supply-reasoning200-v1', task=dataset, name=dataset, model='diffa',
                seed=42, max_new_tokens=512, recipe=original['recipe'], question_order=ids,
                bank={q:original['bank'][q] for q in ids}, records=selected, bare_answers=baselines,
                prompt=PROMPT, bare_prompt=original['prompt'], parent_config_sha256=winners.digest(config),
                primary_metric=PRIMARY)
    winners.freeze(output/'run_config.json',plan)
    return plan


SUPPLY_ONLY_PROTOCOL = 'diffa-multiarith-supply-only200-v1'


def prepare_supply_only(output, audio_root=None):
    from phi4_more_bare_eval import select

    config = output/'run_config.json'
    if config.exists():
        plan = json.loads(config.read_text())
        if (plan['protocol'] != SUPPLY_ONLY_PROTOCOL or plan['task'] != 'multiarith'
                or plan['prompt'] != PROMPT or plan['recipe'] != night.vanilla.RECIPE):
            raise ValueError(f'Incompatible supply-only run in {output}.')
        return plan
    bank, ids, recordings, source = select('multiarith', 200, audio_root=audio_root, sample_seed=42)
    plan = dict(protocol=SUPPLY_ONLY_PROTOCOL, task='multiarith', name='multiarith', model='diffa',
                seed=42, max_new_tokens=512, recipe=night.vanilla.RECIPE, question_order=ids,
                bank=bank, records=recordings,
                prompt=PROMPT, primary_metric=PRIMARY, audio_source=source)
    winners.freeze(config, plan)
    return plan


def report(output, plan):
    new = evaluation.raw_rows(output,plan)
    ids = [q for q in plan['question_order'] if all((q,s) in new for s in STYLES)]
    has_bare = 'bare_answers' in plan
    scope = 'all four conditions' if has_bare else 'both supply-reasoning conditions'
    print(f'{DATASETS[plan["task"]][0]}: {len(new)}/400 new answers; {len(ids)}/200 questions complete in {scope}.')
    if not ids:
        return
    bare = {(r['question_id'],r['style']):r for r in plan.get('bare_answers', [])}
    rows = []
    for q in ids:
        for style in STYLES:
            if has_bare:
                rows.append(bare[q,style])
            rows.append(dict(new[q,style],condition=style+'-reasoning'))
    scored, receipt = winners.score_rows(dict(target=plan),rows,output/'scoring')
    groups = {}
    for row in scored:
        groups.setdefault(row['condition'], {})[row['question_id']] = row
    lengths, comparisons, interactions = {}, [], []
    for c, group in groups.items():
        tokens = [r['response_tokens'] for r in group.values()]
        lengths[c] = dict(mean_tokens=mean(tokens),median_tokens=median(tokens))
        print(f'{c}: mean/median tokens {mean(tokens):.1f}/{median(tokens):g}')
    prompts = [('supply','-reasoning')]
    if has_bare:
        prompts.append(('bare',''))
    for metric in scored[0]['metrics']:
        for prompt, suffix in prompts:
            a,b = groups['neutral'+suffix],groups['angry'+suffix]
            stats = paired([(a[q]['metrics'][metric],b[q]['metrics'][metric]) for q in ids])
            comparisons.append(dict(prompt=prompt,metric=metric,**stats))
            print(f'{prompt} / {metric}: neutral {stats["baseline_correct"]}/{len(ids)} -> '
                  f'angry {stats["steered_correct"]}/{len(ids)}; gain={stats["gain_pp"]:+.2f} pp; '
                  f'p={stats["mcnemar_exact_p"]:.4g}; CI={stats["ci95_gain_pp"]}')
        if not has_bare:
            continue
        result = accuracy_interaction(groups,metric)
        interactions.append(result)
        print(f'{metric} / change in angry-neutral gap (supply minus bare): '
              f'{result["interaction_pp"]:+.2f} pp; p={result["p"]}; CI={result["ci95_pp"]}')
    winners.atomic_text(output/'rescored.jsonl',''.join(json.dumps(r)+'\n' for r in scored))
    winners.atomic_text(output/'summary.json',json.dumps(dict(matched=len(ids),primary_metric=PRIMARY,
        comparisons=comparisons,interactions=interactions,lengths=lengths,scorer=receipt,
        inference='Unadjusted matched McNemar p and paired bootstrap CI for each prompt. '
                  + ('Paired question-level t approximation for the change in gap.' if has_bare else '')),indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('run','prepare','summary'),default='run')
    parser.add_argument('--dataset', choices=DATASETS, default='gsm8k')
    parser.add_argument('--supply-only', action='store_true', help='Generate and score only neutral and angry with the reasoning prompt on MultiArith.')
    parser.add_argument('--audio-root', type=Path, help='Directory containing the Jenny audio inventory.')
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    label, default_parent, default_output = DATASETS[args.dataset]
    if args.supply_only:
        if args.dataset != 'multiarith' or args.parent is not None:
            parser.error('--supply-only uses MultiArith audio and takes no --parent')
        default_output = Path('outputs/diffa-multiarith-supply-only200')
    parent = args.parent or default_parent
    output = args.output or default_output
    if args.mode == 'summary':
        config = output/'run_config.json'
        if config.exists():
            plan = json.loads(config.read_text())
            if plan['task'] != args.dataset or args.supply_only != (plan['protocol'] == SUPPLY_ONLY_PROTOCOL):
                raise ValueError(f'Expected {label} results in {output}.')
            report(output,plan)
        else:
            print(f'{label} supply-reasoning run has not been prepared yet.')
        return
    output.mkdir(parents=True,exist_ok=True)
    with (output/'runner.lock').open('a') as own:
        try:
            fcntl.flock(own,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'{label} supply-reasoning run is already active.')
        plan = prepare_supply_only(output,args.audio_root) if args.supply_only else prepare(parent,output,args.dataset)
        print(f'200 {label} questions, neutral and angry. Generation prompt:\n'+PROMPT,flush=True)
        winners.score_rows(dict(target=plan),[],output/'scoring')
        done = evaluation.raw_rows(output,plan,repair=True)
        pending = [r for r in plan['records'] if (r['question_id'],r['condition']) not in done]
        registry = None
        for i,r in enumerate(pending):
            if not Path(r['audio_file']).is_file():
                registry = registry or night.Registry(night.audio_root(args.audio_root))
                pending[i] = registry.relocate(r)
            if winners.digest(pending[i]['audio_file']) != r['audio_sha256']:
                raise ValueError(f'Audio changed for {r["question_id"]}/{r["style"]}.')
        if args.mode == 'prepare':
            return
        if pending:
            with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
                print('Waiting for the shared DIFFA GPU lock and idle GPU.',flush=True)
                fcntl.flock(gpu,fcntl.LOCK_EX)
                while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                    time.sleep(30)
                from diffa_model_receipts import model_revisions
                from src.dataloader import build_chat_prompt
                packages = {k:version(k) for k in ('torch','transformers')}
                revisions = model_revisions()
                parent_runtime = parent/'runtime.json'
                if not args.supply_only and parent_runtime.exists():
                    old = json.loads(parent_runtime.read_text())
                    if old['packages'] != packages or old['model_revisions'] != revisions:
                        raise ValueError('Model revisions or package versions differ from the bare run.')
                backend = night.DiffaBackend()
                text, count = evaluation.context_guard(backend.tokenizer,PROMPT,build_chat_prompt)
                winners.atomic_text(output/'native-chat-prompt.txt',text)
                runtime_path = output/'runtime.json'
                runner_digest = winners.digest(__file__)
                receipt = dict(packages=packages,model_revisions=revisions,
                    model_config=backend.model.config.to_dict(),native_prompt_tokens=count,
                    runner_sha256=runner_digest)
                if not args.supply_only and runtime_path.exists():
                    previous_digest = json.loads(runtime_path.read_text()).get('runner_sha256')
                    legacy = {'25ea2dce83c38df58393fad861fe7399de6a535020b62f35ffc4809f71d8283e'}
                    if args.dataset == 'gsm8k':
                        legacy.add(GSM_RUNNER_V1)
                    if previous_digest in legacy:
                        winners.freeze(runtime_path, dict(receipt, runner_sha256=previous_digest))
                        runtime_path = output/f'runtime-resumed-{runner_digest[:12]}.json'
                winners.freeze(runtime_path, receipt)
                evaluation.generate_dataset(output,plan,pending,backend)
        report(output,plan)


if __name__ == '__main__':
    main()
