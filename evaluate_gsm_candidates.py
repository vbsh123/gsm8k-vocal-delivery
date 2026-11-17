"""Full 1269-question GSM8K neutral/angry evaluation with native candidate audio models."""
import argparse
import fcntl
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

from bbh_candidate_adapters import MODELS
import evaluate_bbh_candidates as candidates
from kimi_bbh_layer_diagnostic import digest, freeze
from run_kimi_gsm8k import QUESTION_FILE, prepare_audio
import run_reasoning_transfer as reasoning
from steer import read_rows

EXPLICIT = reasoning.HASH_PROMPT
ORIGINAL = EXPLICIT.replace('Give concise step-by-step reasoning with only the necessary calculations.',
                            "Let's think step by step.")


def questions():
    rows = json.loads(QUESTION_FILE.read_text())['questions']
    if len(rows) != 1269 or len({r['question_id'] for r in rows}) != 1269:
        raise ValueError('Expected all 1269 main-cohort GSM8K questions.')
    random.Random(42).shuffle(rows)
    return rows


def audio(args):
    rows = questions(); args.source.mkdir(parents=True, exist_ok=True)
    reasoning.bank_file(args.source/'questions.jsonl', rows)
    if (args.source/'plan.jsonl').exists():
        try:
            make_plan(args.source, args.model, 'explicit')
        except FileNotFoundError:
            pass
        else:
            print('All 2538 neutral/angry GSM8K clips verified; reusing existing audio.', flush=True)
            return
    prepare_audio(args, dict(collection=[], test=rows))


def make_plan(source, model, variant):
    if model not in MODELS or variant not in ('original', 'explicit'):
        raise ValueError('Unknown model/prompt.')
    rows = questions(); bank = {r['question_id']: r for r in rows}
    if read_rows(source/'questions.jsonl') != rows:
        raise ValueError('GSM8K source question bank differs from the frozen 1269-question cohort.')
    records = reasoning.source_records(source, bank, ['neutral', 'angry'])
    tasks = []
    for q in rows:
        neutral, angry = (records[q['question_id'], s] for s in ('neutral', 'angry'))
        if neutral['voice'] != angry['voice']:
            raise ValueError('Question voices differ between deliveries.')
        for record in (neutral, angry):
            task = dict(record, condition=record['style'], reference_answer=record['answer'])
            task.pop('answer'); tasks.append(task)
    if model == 'flamingo' and any(r['seconds'] > 600 for r in tasks):
        raise ValueError('AF3 cannot process this full recording without truncation.')
    return dict(task='gsm8k', protocol='gsm8k-1269-candidate-delivery-v1', model_key=model,
        source=str(source.absolute()), question_file_sha256=digest(QUESTION_FILE), bank=bank,
        recordings=tasks, prompt=EXPLICIT if variant == 'explicit' else ORIGINAL, prompt_variant=variant,
        recipe=dict(MODELS[model], dtype='bfloat16', do_sample=False, num_beams=1, use_cache=True,
                    repetition_penalty=1., text_only_output=True), seed=42, max_new_tokens=512,
        selection='All 1269 main Azure cohort questions; no collection or steering, outcome/classifier/duration filtering.',
        interventions=[], audio_encoding='Full mono 16kHz waveform through native candidate processor/encoder.',
        duration_audit=dict(max_seconds=max(r['seconds'] for r in tasks), over_40_seconds=sum(r['seconds'] > 40 for r in tasks)),
        scoring='Installed upstream GSM8K filters and exact-match metric; numeric equivalence is additional. '
                'Audio-adapted inference, not full official text-harness prompting.')


def prepare(args):
    plans = []
    for variant in ('original', 'explicit') if args.prompt == 'both' else (args.prompt,):
        output = args.output/args.model/variant; output.mkdir(parents=True, exist_ok=True)
        plan = make_plan(args.source, args.model, variant)
        freeze(output/'run_config.json', plan); plans.append((output, plan))
        print(f'{args.model}/{variant}: all 1269 GSM8K questions, 2538 neutral/angry answers; no steering.', flush=True)
    return plans


def report(output):
    from rescore_audio_steering import HarnessTaskScorer, numeric_equal, paired
    path = output/'run_config.json'
    if not path.exists():
        print(f'{output}: not prepared.'); return
    plan = json.loads(path.read_text()); rows = read_rows(output/'results.jsonl')
    if not rows:
        print(f'{output}: no saved answers yet.'); return
    for row in rows:
        if row['prompt'] != plan['prompt'] or row['reference_answer'] != plan['bank'][row['question_id']]['answer']:
            raise ValueError('Saved prompt/reference differs from frozen GSM8K protocol.')
    scorer = HarnessTaskScorer('gsm8k', output/'harness/upstream')
    docs = [dict(question=plan['bank'][r['question_id']]['question'], answer=r['reference_answer']) for r in rows]
    values = scorer.score([r['response'] for r in rows], docs)
    scored = [dict(row, scores={**{m: v[i]['correct'] for m, v in values.items()},
        'flexible-numeric': numeric_equal(values['flexible-extract'][i]['prediction'], row['reference_answer']),
        'format-compliant': values['strict-match'][i]['prediction'] != '[invalid]'}) for i, row in enumerate(rows)]
    groups = {s: {} for s in ('neutral', 'angry')}
    for row in scored:
        key = row['question_id'], row['voice']; group = groups[row['style']]
        if key in group and group[key]['response'] != row['response']:
            raise ValueError('Conflicting answers.')
        group[key] = row
    keys = sorted(groups['neutral'].keys() & groups['angry'].keys()); n = len(keys)
    print(f'\n{plan["recipe"]["model"]} GSM8K ({plan["prompt_variant"]}): {len(rows)}/{len(plan["recordings"])} answers; {n} matched questions.')
    accuracy, comparisons = [], []
    if n:
        print('Delivery     N   Numeric  Flexible   Strict   Format   Mean tokens  Median tokens')
        for style in groups:
            counts = {m: sum(groups[style][k]['scores'][m] for k in keys)
                      for m in ('flexible-numeric', 'flexible-extract', 'strict-match', 'format-compliant')}
            lengths = [groups[style][k]['response_tokens'] for k in keys]
            accuracy.append(dict(style=style, n=n, correct=counts, mean_tokens=statistics.mean(lengths), median_tokens=statistics.median(lengths)))
            print(f'{style:8} {n:5d} '+ ' '.join(f'{counts[m]/n:8.1%}' for m in counts)
                  +f' {statistics.mean(lengths):12.1f} {statistics.median(lengths):14.1f}')
        for metric in ('flexible-numeric', 'flexible-extract', 'strict-match', 'format-compliant'):
            result = paired([(groups['neutral'][k]['scores'][metric], groups['angry'][k]['scores'][metric]) for k in keys])
            comparisons.append(dict(metric=metric, **result))
            print(f'{metric} / angry: gain={result["gain_pp"]:+.2f} pp; W/L={result["wins"]}/{result["losses"]}; '
                  f'p={result["mcnemar_exact_p"]:.4f}; CI={result["ci95_gain_pp"]}')
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    (output/'summary.json').write_text(json.dumps(dict(primary_metric='flexible-numeric', accuracy=accuracy,
        paired=comparisons, scorer=scorer.provenance, prompt=plan['prompt'], prompt_variant=plan['prompt_variant'],
        note=plan['scoring']+' Matched questions only; p-values/intervals unadjusted; prompt variants are not independent cohorts.'), indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=tuple(MODELS), default='phi4')
    p.add_argument('--mode', choices=('audio', 'prepare', 'evaluate', 'summary'), default='evaluate')
    p.add_argument('--prompt', choices=('original', 'explicit', 'both'), default='both')
    p.add_argument('--source', type=Path, default=Path('outputs/azure-kimi-gsm8k1269'))
    p.add_argument('--output', type=Path, default=Path('outputs/gsm8k-candidates'))
    p.add_argument('--audio-sources', nargs='+', type=Path, default=[Path('outputs')/s for s in
        ('azure-jenny500-extra', 'azure-jenny500-neutral', 'azure-jenny-next500', 'azure-jenny-remaining-test')])
    p.add_argument('--max-generations', type=int, default=0)
    args = p.parse_args()
    if args.max_generations < 0:
        p.error('--max-generations must be nonnegative')
    if args.mode == 'audio':
        audio(args); return
    if args.mode == 'summary':
        for variant in ('original', 'explicit') if args.prompt == 'both' else (args.prompt,):
            report(args.output/args.model/variant)
        return
    plans = prepare(args)
    if args.mode == 'prepare': return
    with (args.output/args.model/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            candidates.run(plans, args.model, args.max_generations)


if __name__ == '__main__':
    main()
