"""BBH Object Counting: neutral reasoning + four bare-task natural deliveries.

Azure audio generation followed by DIFFA natural evaluation only. No examples,
activation collection, steering, emotion classification or outcome filtering.
"""
import argparse
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import diffa_fewshot_emotions as evaluation
from rescore_audio_steering import HarnessTaskScorer, download, paired
from run_reasoning_transfer import bank_file, source_records
from steer import read_rows
from statistics import mean, median

TASK = 'object_counting'
REVISION = '9ee07bd481feebf959a6b59d61ea57bdcf30964d'
# Git blob SHA from the authors' immutable repository tree, including exact bytes.
SOURCE_BLOB = 'f94e49dd473e2e6a5b6be7da4cc9c9cf8c9da693'
SOURCE_URL = f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{REVISION}/bbh/{TASK}.json'
PROTOCOL = 'diffa-object-counting-five-conditions-v1'
BARE_PROMPT = ('Answer the question in the audio. End your response with a final answer line in the format:\n'
               'The answer is <number>.')
REASONING_SENTENCE = 'Give concise step-by-step reasoning with only the necessary calculations.'
REASONING_PROMPT = BARE_PROMPT.replace('audio. ', 'audio. '+REASONING_SENTENCE+' ', 1)
VOICE = 'en-US-JennyNeural'
STYLES = ('neutral', 'angry', 'happy', 'sad')
CONDITIONS = ('neutral-reasoning', *STYLES)
PROMPTS = {c: REASONING_PROMPT if c == 'neutral-reasoning' else BARE_PROMPT for c in CONDITIONS}
winners = evaluation.winners


def source_questions(data):
    digest = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if digest != SOURCE_BLOB:
        raise ValueError('Object Counting source differs from the pinned authors\' Git blob.')
    examples = json.loads(data)['examples']
    if len(examples) != 250 or any(not x['target'].isdigit() for x in examples):
        raise ValueError('Expected all250 official Object Counting questions with numeric targets.')
    order = list(enumerate(examples))
    random.Random(42).shuffle(order)
    return [dict(question_id=f'bbh_object_counting_{i:03d}', source_index=i,
                 original_input=x['input'], original_target=x['target'],
                 question=x['input'], answer=x['target']) for i, x in order]


def prepare(source, output):
    source.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    path = source/'bbh-source.json'
    data = path.read_bytes() if path.exists() else download(SOURCE_URL)
    questions = source_questions(data)
    if not path.exists():
        path.write_bytes(data)
    bank_file(source/'questions.jsonl', questions)
    selection = dict(protocol=PROTOCOL, name=TASK, model='diffa', task=TASK,
        label='five-condition audio prompt comparison', prompt=BARE_PROMPT, prompts=PROMPTS,
        source_url=SOURCE_URL, source_git_blob=SOURCE_BLOB, source_sha256=hashlib.sha256(data).hexdigest(),
        bank={q['question_id']: q for q in questions}, question_order=[q['question_id'] for q in questions],
        styles=list(STYLES), conditions=list(CONDITIONS), voice=VOICE, azure_intensity=1.5,
        recipe=dict(evaluation.night.vanilla.RECIPE), seed=42, max_new_tokens=512,
        primary_metric='flexible-extract',
        adaptation='Custom audio prompts: neutral audio with the original math reasoning sentence; '
                   'the same neutral audio plus angry/happy/sad audio with the bare prompt. '
                   'All250 question texts spoken unchanged in four styles. Only the reasoning sentence differs '
                   'between prompts. Native DIFFA decoder/chat wrapping; installed harness saved-answer filters '
                   'and exact_match scoring. No examples, E2V/accuracy filtering, fitting or steering.')
    winners.freeze(output/'selection.json', selection)
    winners.atomic_text(output/'prompt-bare.txt', BARE_PROMPT+'\n')
    winners.atomic_text(output/'prompt-reasoning.txt', REASONING_PROMPT+'\n')
    return selection


def synthesize(source, prepare_only=False):
    command = [sys.executable, '-u', 'cloud_tts_smoke.py', '--provider', 'azure',
               '--azure-voice', VOICE, '--azure-intensity', '1.5',
               '--input-jsonl', str(source/'questions.jsonl'), '--num-examples', '250',
               '--styles', *STYLES, '--output', str(source), '--resume']
    if prepare_only:
        command.append('--prepare-only')
    subprocess.run(command, check=True)


def plan_for(selection, source):
    records = source_records(source, selection['bank'], STYLES)
    return dict(selection, records=[dict(records[q, 'neutral' if c == 'neutral-reasoning' else c],
                                        condition=c, reference_answer=selection['bank'][q]['answer'])
                                   for q in selection['question_order'] for c in CONDITIONS])


def scoring_preflight(output):
    scorer = HarnessTaskScorer('bbh_cot_zeroshot_object_counting', output/'scoring')
    values = scorer.score([f'The answer is {n}.' for n in (2, 8, 18)],
                          [dict(input='Format check only', target=str(n)) for n in (2, 8, 18)])
    if not all(v['correct'] for group in values.values() for v in group):
        raise ValueError('Agreed final-answer format does not pass the installed upstream checkers.')
    return scorer


def raw_rows(output, plan, repair=False):
    expected = {(r['question_id'], r['condition']): r for r in plan['records']}
    checksum = winners.digest(output/'run_config.json')
    done = {}
    for row in read_rows(output/'deliveries.jsonl', repair=repair):
        key = row['question_id'], row['condition']
        record = expected.get(key)
        if (key in done or record is None or row.get('run_config_sha256') != checksum
                or row.get('prompt') != plan['prompts'][key[1]] or row.get('model') != 'diffa'
                or row.get('interventions') != [] or row.get('strength') != 0.
                or any(row.get(k) != v for k, v in plan['recipe'].items())
                or any(row.get(k) != record[k] for k in ('voice', 'style', 'audio_sha256', 'reference_answer'))):
            raise ValueError('Saved answer differs from frozen condition/prompt/audio/decoder, or is duplicated.')
        done[key] = row
    return done


def generate(plan, output, backend):
    done = raw_rows(output, plan, repair=True)
    checksum = winners.digest(output/'run_config.json')
    with (output/'deliveries.jsonl').open('a') as stream:
        for record in plan['records']:
            key = record['question_id'], record['condition']
            if key in done:
                continue
            prompt = plan['prompts'][record['condition']]
            print(f'Object Counting [{len(done)+1}/{len(plan["records"])}] {key[0]}/{key[1]}', flush=True)
            trace = backend.generate(record, dict(plan, prompt=prompt), capture=False)
            row = dict(record, prompt=prompt, response=trace['response'], response_tokens=trace['response_tokens'],
                       audio_encoding=trace['audio_encoding'], model='diffa', strength=0., interventions=[],
                       run_config_sha256=checksum, **plan['recipe'])
            stream.write(json.dumps(row)+'\n'); stream.flush(); done[key] = row


def report(output):
    plan = json.loads((output/'run_config.json').read_text())
    scorer = HarnessTaskScorer('bbh_cot_zeroshot_object_counting', output/'scoring')
    rows = list(raw_rows(output, plan).values())
    docs = [dict(input=plan['bank'][r['question_id']]['original_input'],
                 target=plan['bank'][r['question_id']]['original_target']) for r in rows]
    values = scorer.score([r['response'] for r in rows], docs)
    scored = [dict(r, metrics={m: v[i]['correct'] for m, v in values.items()},
                   predictions={m: v[i]['prediction'] for m, v in values.items()}) for i, r in enumerate(rows)]
    groups = {c: {r['question_id']: r for r in scored if r['condition'] == c} for c in CONDITIONS}
    print(f'\nObject Counting: {len(rows)}/{len(plan["records"])} answers; five conditions per question.', flush=True)
    accuracy, lengths, comparisons = [], {}, []
    for condition, group in groups.items():
        if not group:
            continue
        tokens = [r['response_tokens'] for r in group.values()]
        lengths[condition] = dict(n=len(tokens), mean_tokens=mean(tokens), median_tokens=median(tokens))
        print(f'{condition}: N={len(group)}; mean/median tokens={mean(tokens):.1f}/{median(tokens):g}', flush=True)
        for metric in values:
            correct = sum(r['metrics'][metric] for r in group.values())
            accuracy.append(dict(condition=condition, metric=metric, n=len(group), correct=correct))
            print(f'  {metric}: {correct}/{len(group)} ({correct/len(group):.1%})', flush=True)
    for test in ('neutral-reasoning', 'angry', 'happy', 'sad'):
        ids = sorted(groups['neutral'].keys() & groups[test].keys())
        if not ids:
            continue
        matched_lengths = {c: dict(mean_tokens=mean(groups[c][q]['response_tokens'] for q in ids),
                                  median_tokens=median(groups[c][q]['response_tokens'] for q in ids))
                           for c in ('neutral', test)}
        for metric in values:
            stats = paired([(groups['neutral'][q]['metrics'][metric], groups[test][q]['metrics'][metric]) for q in ids])
            comparisons.append(dict(reference='neutral', test=test, metric=metric,
                                    matched_lengths=matched_lengths, **stats))
            print(f'{test} vs neutral-bare / {metric}: N={len(ids)}; gain={stats["gain_pp"]:+.2f} pp; '
                  f'p={stats["mcnemar_exact_p"]:.4g}; CI={stats["ci95_gain_pp"]}', flush=True)
    winners.atomic_text(output/'delivery-rescored.jsonl', ''.join(json.dumps(r)+'\n' for r in scored))
    winners.atomic_text(output/'delivery-summary.json', json.dumps(dict(protocol=PROTOCOL,
        accuracy=accuracy, lengths=lengths, comparisons=comparisons, scorer=scorer.provenance,
        adaptation=plan['adaptation'],
        statistics='Exact paired McNemar p, paired bootstrap95% CI; unadjusted. Partial comparisons use matched IDs.'), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'prepare', 'evaluate', 'summary'), default='run')
    parser.add_argument('--source', type=Path, default=Path('outputs/azure-object-counting250'))
    parser.add_argument('--output', type=Path, default=Path('outputs/diffa-object-counting-prompts'))
    args = parser.parse_args()
    config = args.output/'run_config.json'
    if args.mode == 'summary':
        if config.exists():
            report(args.output)
        else:
            clips = [p for p in args.source.glob('*.wav') if p.stat().st_size > 44]
            print(f'Object Counting: audio {len(clips)}/1000 clips; evaluation not prepared yet.', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'runner.lock').open('a') as own:
        try:
            fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Object Counting runner already active; use --mode summary to inspect it.')
        scorer = scoring_preflight(args.output)
        if args.mode in ('run', 'prepare'):
            selection = prepare(args.source, args.output)
            print('Object Counting: all250 questions × five conditions; 1000 audio clips, 1250 generations.', flush=True)
            print('Bare prompt:\n'+BARE_PROMPT+'\nNeutral reasoning prompt:\n'+REASONING_PROMPT, flush=True)
            synthesize(args.source, prepare_only=args.mode == 'prepare')
            if args.mode == 'prepare':
                print('Prepared only; no cloud synthesis or GPU inference.', flush=True)
                return
            plan = plan_for(selection, args.source)
            winners.freeze(config, plan)
        else:
            plan = json.loads(config.read_text())
            if plan['protocol'] != PROTOCOL or plan['prompts'] != PROMPTS:
                raise ValueError('Frozen Object Counting condition differs.')
        done = raw_rows(args.output, plan)
        if len(done) < len(plan['records']):
            for path, expected in {r['audio_file']: r['audio_sha256'] for r in plan['records']}.items():
                if winners.digest(path) != expected:
                    raise ValueError('Audio changed after preparation.')
            with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
                print('Waiting for shared GPU lock and idle GPU; audio generation is already complete.', flush=True)
                fcntl.flock(gpu, fcntl.LOCK_EX)
                while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                    time.sleep(30)
                backend = evaluation.night.DiffaBackend()
                from src.dataloader import build_chat_prompt
                from diffa_model_receipts import model_revisions
                contexts = {}
                for label, prompt in (('bare', BARE_PROMPT), ('reasoning', REASONING_PROMPT)):
                    text, count = evaluation.context_guard(backend.tokenizer, prompt, build_chat_prompt)
                    winners.atomic_text(args.output/f'native-chat-prompt-{label}.txt', text)
                    contexts[label] = count
                winners.freeze(args.output/'runtime.json', dict(model_revisions=model_revisions(),
                    packages={k: version(k) for k in ('torch', 'transformers')},
                    model_config=backend.model.config.to_dict(), native_prompt_tokens=contexts,
                    runner_sha256=winners.digest(__file__)))
                generate(plan, args.output, backend)
        report(args.output)


if __name__ == '__main__':
    main()
