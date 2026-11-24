"""DIFFA pilot: speak the instruction and supply the math question in writing."""
import argparse
import json
from pathlib import Path
from statistics import mean, median

import diffa_fewshot_emotions as evaluation
from phi4_queued_workers import gpu_queue
from phi4_sweep import job_lock
from run_arithmetic_audio_pilot import accuracy_interaction
from run_reasoning_transfer import generate_audio, source_records
from rescore_audio_steering import paired
from steer import read_rows

winners = evaluation.winners
DATA = Path(__file__).resolve().parent / 'data/spoken_prompt_math50.json'
STYLES = ('neutral', 'angry')
SPOKEN = {
    'bare': 'Solve the written math problem. End your response with a final answer line in this format: four hash signs, a space, then the number.',
    'original': 'Solve the written math problem. Give concise step-by-step reasoning with only the necessary calculations. If the question asks for a total, compute that total. End with one plain-text line in exactly this format: four hash signs, a space, then the number. Put a single numeric answer on that line, without Markdown, units, or separate component counts.',
}


def saved(folder, plan, repair=False):
    expected = {(r['question_id'], r['condition']): r for r in plan['records']}
    checksum = winners.digest(folder / 'run_config.json')
    rows = {}
    for r in read_rows(folder / 'deliveries.jsonl', repair=repair):
        key = r['question_id'], r['condition']
        record = expected[key]
        if key in rows or r['run_config_sha256'] != checksum or any(r[k] != v for k,v in record.items()):
            raise ValueError(f'Incompatible or duplicate saved delivery: {key}')
        rows[key] = r
    return rows


def prepare(folder, task, dataset, clips):
    bank = {q['question_id']:q for q in dataset['questions']}
    records = []
    for qid, q in bank.items():
        for template, speech in SPOKEN.items():
            for style in STYLES:
                clip = clips[template, style]
                records.append(dict(question_id=qid, condition=f'{template}-{style}', style=style,
                    instruction=template, spoken_prompt=speech, prompt=q['question'], reference_answer=q['answer'],
                    audio_file=clip['audio_file'], audio_sha256=clip['audio_sha256'], seconds=clip['seconds']))
    plan = dict(protocol='diffa-spoken-instruction-math-v1', task=task, name=task, model='diffa',
                bank=bank, records=records, recipe=evaluation.night.vanilla.RECIPE,
                question_set_sha256=winners.digest(DATA), source_plan_sha256=dataset['source_plan_sha256'],
                original_spoken_question_numeric=dataset['original_spoken_question_numeric'],
                original_scores_sha256=dataset['original_scores_sha256'],
                original_written_instruction=dataset['original_prompt'], max_new_tokens=512)
    winners.freeze(folder / 'run_config.json', plan)
    return plan


def generate(folder, plan, backend, builder):
    done = saved(folder, plan, repair=True)
    checksum = winners.digest(folder / 'run_config.json')
    with (folder / 'deliveries.jsonl').open('a') as stream:
        for record in plan['records']:
            key = record['question_id'], record['condition']
            if key in done:
                continue
            evaluation.context_guard(backend.tokenizer, record['prompt'], builder)
            trace = backend.generate(record, dict(plan, prompt=record['prompt']), capture=False)
            row = dict(record, **trace, model='diffa', run_config_sha256=checksum, **plan['recipe'])
            stream.write(json.dumps(row)+'\n')
            stream.flush()
            done[key] = row
            print(f'{plan["name"]} {len(done)}/{len(plan["records"])} {key[0]} {key[1]} '
                  f'{row["response_tokens"]} tokens', flush=True)


def report(folder):
    config = folder / 'run_config.json'
    if not config.is_file():
        print(f'{folder.name}: awaiting preparation.')
        return
    plan = json.loads(config.read_text())
    rows = list(saved(folder, plan).values())
    print(f'{folder.name}: {len(rows)}/{len(plan["records"])} answers.')
    if not rows:
        return
    scored, provenance = winners.score_rows(dict(target=plan), rows, folder/'scoring')
    groups = {f'{p}-{s}':{} for p in SPOKEN for s in STYLES}
    for r in scored:
        groups[r['condition']][r['question_id']] = r
    results = {}
    for p in SPOKEN:
        n, a = (groups[f'{p}-{s}'] for s in STYLES)
        ids = sorted(n.keys() & a.keys())
        if not ids:
            continue
        lengths = [[group[q]['response_tokens'] for q in ids] for group in (n,a)]
        print(f'{p}: {len(ids)} matched. Neutral / angry mean (median) tokens '
              + ' / '.join(f'{mean(v):.1f} ({median(v):g})' for v in lengths))
        results[p] = {}
        for metric in n[ids[0]]['metrics']:
            st = paired([(n[q]['metrics'][metric],a[q]['metrics'][metric]) for q in ids])
            results[p][metric] = st
            print(f'  {metric}: {st["baseline_correct"]}/{len(ids)} -> {st["steered_correct"]}/{len(ids)}; '
                  f'{st["gain_pp"]:+.2f} pp; 95% CI {st["ci95_gain_pp"]}; p={st["mcnemar_exact_p"]:.4g}')
    interaction_groups = {s+suffix:groups[f'{p}-{s}'] for p,suffix in [('bare',''),('original','-reasoning')] for s in STYLES}
    interaction = accuracy_interaction(interaction_groups, 'flexible-numeric')
    if interaction:
        print('Change in angry-neutral numeric gap (original minus bare): '
              f'{interaction["interaction_pp"]:+.2f} pp on {interaction["pairs"]} questions; '
              f'95% CI {interaction["ci95_pp"]}; p={interaction["p"]}')
    modalities = {}
    for style in STYLES:
        modalities[style] = {qid:dict(metrics={'flexible-numeric':correct})
                             for qid,correct in plan['original_spoken_question_numeric'][style].items()}
        modalities[style+'-reasoning'] = groups['original-'+style]
    modality_change = accuracy_interaction(modalities, 'flexible-numeric')
    if modality_change:
        print('Original instruction: change in angry-neutral numeric gap when moving speech '
              f'from question to instruction: {modality_change["interaction_pp"]:+.2f} pp '
              f'on {modality_change["pairs"]} questions; 95% CI {modality_change["ci95_pp"]}; p={modality_change["p"]}')
    winners.atomic_text(folder/'rescored.jsonl',''.join(json.dumps(r)+'\n' for r in scored))
    winners.atomic_text(folder/'summary.json',json.dumps(dict(comparisons=results,interaction=interaction,
        original_modality_change=modality_change,scorer=provenance),indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=('gsm8k','multiarith'), default=['gsm8k','multiarith'])
    parser.add_argument('--output', type=Path, default=Path('outputs/diffa-spoken-prompt-math50'))
    parser.add_argument('--mode', choices=['run','summary'], default='run')
    args = parser.parse_args()
    if args.mode == 'summary':
        for task in args.datasets:
            report(args.output/task)
        return
    with job_lock(args.output):
        datasets = json.loads(DATA.read_text())['datasets']
        audio = args.output/'audio'
        prompts = {k:dict(question_id=k,question=v,answer='') for k,v in SPOKEN.items()}
        generate_audio(audio, list(prompts.values()), STYLES)
        clips = source_records(audio, prompts, STYLES)
        plans = {task:prepare(args.output/task,task,datasets[task],clips) for task in args.datasets}
        pending = [task for task,plan in plans.items() if len(saved(args.output/task,plan,repair=True)) < len(plan['records'])]
        if pending:
            with gpu_queue():
                from src.dataloader import build_chat_prompt
                from diffa_model_receipts import model_revisions
                backend = evaluation.night.DiffaBackend()
                winners.freeze(args.output/'model-revisions.json', model_revisions())
                for task in pending:
                    generate(args.output/task, plans[task], backend, build_chat_prompt)
                    report(args.output/task)
        for task in plans.keys()-set(pending):
            report(args.output/task)


if __name__ == '__main__':
    main()
