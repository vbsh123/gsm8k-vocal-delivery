"""DIFFA supply-reasoning pilot on spoken BBH Tracking Shuffled Objects (7) and Navigate."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import re
from statistics import mean, median

import diffa_fewshot_emotions as evaluation
from phi4_queued_workers import gpu_queue
from phi4_sweep import job_lock
from rescore_audio_steering import HarnessTaskScorer, download, paired
from run_arithmetic_audio_pilot import REVISION, REASONING_SENTENCE
from run_reasoning_transfer import bank_file, generate_audio, source_records

winners = evaluation.winners
TASKS = {'shuffled7': 'tracking_shuffled_objects_seven_objects', 'navigate': 'navigate'}
SOURCE_HASHES = {'shuffled7': 'b59fd05b850c37cf57cb514ae1e6975c00b1638ade2291722056b2af3c54084d',
                 'navigate': 'b7944dfd3c81da5c4815bd75b00d6fa3c64140823498e78ed496f4c3d36f7f0d'}
STYLES = ('angry', 'neutral')
ENDING = {'shuffled7': 'The answer is (<letter>).\nReplace <letter> with A, B, C, D, E, F, or G.',
          'navigate': 'The answer is <Yes or No>.\nReplace <Yes or No> with Yes or No.'}


def prepare(task, folder):
    source = folder/'audio'
    source.mkdir(parents=True, exist_ok=True)
    path = source/'bbh-source.json'
    url = f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{REVISION}/bbh/{TASKS[task]}.json'
    data = path.read_bytes() if path.exists() else download(url)
    if hashlib.sha256(data).hexdigest() != SOURCE_HASHES[task]:
        raise ValueError(f'Unexpected {task} source checksum.')
    if not path.exists():
        winners.atomic_text(path, data.decode())
    examples = json.loads(data)['examples']
    order = list(range(len(examples)))
    random.Random(42).shuffle(order)
    bank = {}
    for index in order[:50]:
        example = examples[index]
        stem, options = example['input'].rsplit('\nOptions:\n', 1)
        if task == 'shuffled7':
            spoken = re.sub(r'^\(([A-G])\) ', r'Option \1: ', options, flags=re.MULTILINE)
        else:
            spoken = options.replace('- ', '')
        q = f'bbh_{TASKS[task]}_{index:03d}'
        bank[q] = dict(question_id=q, question=stem+'\nOptions:\n'+spoken, answer=example['target'],
                       original_input=example['input'], original_target=example['target'], source_index=index)
    prompt = 'Answer the question in the audio. '+REASONING_SENTENCE+' End your response with a final answer line in the format:\n'+ENDING[task]
    plan = dict(protocol='diffa-bbh-supply-pilot-v1', name=TASKS[task], task=TASKS[task], model='diffa',
                prompt=prompt, bank=bank, question_order=list(bank), styles=list(STYLES), voice='en-US-JennyNeural',
                azure_intensity=1.5, max_new_tokens=512, seed=42, recipe=evaluation.night.vanilla.RECIPE,
                source_url=url, source_sha256=SOURCE_HASHES[task], primary_metric='flexible-extract')
    winners.freeze(folder/'selection.json', plan)
    winners.atomic_text(folder/'prompt.txt', prompt+'\n')
    bank_file(source/'questions.jsonl', list(bank.values()))
    return plan


def scorer_for(plan, folder):
    return HarnessTaskScorer('bbh_cot_zeroshot_'+plan['task'], folder/'scoring')


def documents(plan, rows):
    return [dict(input=plan['bank'][r['question_id']]['original_input'],
                 target=plan['bank'][r['question_id']]['original_target']) for r in rows]


def report(plan, folder, scorer):
    saved = evaluation.raw_rows(folder, plan)
    ids = [q for q in plan['question_order'] if all((q, style) in saved for style in STYLES)]
    print(f'{plan["task"]}: {len(saved)}/{len(plan["records"])} answers, {len(ids)}/50 matched reasoning pairs.')
    if not ids:
        return
    rows = [saved[q, s] for q in ids for s in ('neutral', 'angry')]
    scores = scorer.score([r['response'] for r in rows], documents(plan, rows))
    lengths, comparisons = {}, {}
    for style in ('neutral', 'angry'):
        tokens = [saved[q, style]['response_tokens'] for q in ids]
        lengths[style] = dict(mean=mean(tokens), median=median(tokens))
        print(f'{style} + reasoning: mean/median tokens {mean(tokens):.1f}/{median(tokens):g}')
    for metric, entries in scores.items():
        comparisons[metric] = stats = paired([(a['correct'], b['correct']) for a, b in zip(entries[::2], entries[1::2])])
        n = len(ids)
        a, b = stats['baseline_correct'], stats['steered_correct']
        lo, hi = stats['ci95_gain_pp']
        print(f'{metric}: neutral {a}/{n} ({a/n:.1%}) -> angry {b}/{n} ({b/n:.1%}); '
              f'gain={stats["gain_pp"]:+.2f} pp; 95% CI [{lo:.2f}, {hi:.2f}]; p={stats["mcnemar_exact_p"]:.4g}')
    scored = [dict(r, scores={m:v[i] for m,v in scores.items()}) for i,r in enumerate(rows)]
    winners.atomic_text(folder/'rescored.jsonl', ''.join(json.dumps(r)+'\n' for r in scored))
    winners.atomic_text(folder/'summary.json', json.dumps(dict(matched=len(ids), lengths=lengths, comparisons=comparisons,
                                                              scorer=scorer.provenance), indent=2)+'\n')


def run(task, folder, mode):
    config = folder/'run_config.json'
    if mode == 'summary':
        if config.exists():
            plan = json.loads(config.read_text())
            report(plan, folder, scorer_for(plan, folder))
        else:
            print(f'{task}: {len(list((folder/"audio").glob("*.wav")))}/100 audio clips, awaiting inference.')
        return
    with job_lock(folder):
        plan = prepare(task, folder)
        scorer = scorer_for(plan, folder)
        questions = list(plan['bank'].values())
        docs = documents(plan, questions)
        checks = scorer.score([f'The answer is {doc["target"]}.' for doc in docs], docs)
        if not all(v['correct'] for entries in checks.values() for v in entries):
            raise ValueError('Final answer format failed the benchmark scoring preflight.')
        print(f'{task}: 50 questions, 100 answers. Prompt:\n{plan["prompt"]}', flush=True)
        if mode == 'prepare':
            return
        source = folder/'audio'
        generate_audio(source, questions, STYLES)
        clips = source_records(source, plan['bank'], STYLES)
        records = [dict(clips[q, s], condition=s, reference_answer=plan['bank'][q]['answer'])
                   for q in plan['question_order'] for s in STYLES]
        plan = dict(plan, records=records)
        winners.freeze(config, plan)
        if len(evaluation.raw_rows(folder, plan, repair=True)) < len(records):
            with gpu_queue():
                from diffa_model_receipts import model_revisions
                from src.dataloader import build_chat_prompt
                backend = evaluation.night.DiffaBackend()
                text, count = evaluation.context_guard(backend.tokenizer, plan['prompt'], build_chat_prompt)
                winners.atomic_text(folder/'native-chat-prompt.txt', text)
                winners.freeze(folder/'runtime.json', dict(model_revisions=model_revisions(), native_prompt_tokens=count,
                    packages={k:version(k) for k in ('torch', 'transformers')}, model_config=backend.model.config.to_dict()))
                evaluation.generate_dataset(folder, plan, records, backend)
        report(plan, folder, scorer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', choices=TASKS, nargs='+', default=list(TASKS))
    parser.add_argument('--mode', choices=('run', 'summary', 'prepare'), default='run')
    parser.add_argument('--output', type=Path, default=Path('outputs/diffa-bbh-supply50'))
    args = parser.parse_args()
    for dataset in args.datasets:
        run(dataset, args.output/dataset, args.mode)
