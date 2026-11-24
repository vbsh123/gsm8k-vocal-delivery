"""Fixed 50-question AQuA-RAT neutral/angry audio pilot; custom bare prompt, no steering."""
import argparse
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import re
from statistics import mean, median
import subprocess
import sys
import time

import run_object_counting_prompts as base
from rescore_audio_steering import download, paired
from run_reasoning_transfer import bank_file, source_records

REVISION = '26b64ed1f22208c6742f47df547dfd65088ec30d'
SOURCE_BLOB = 'a62ef3e8bc0446de096c8384a0bf5c99867043b2'
SOURCE_URL = f'https://raw.githubusercontent.com/google-deepmind/AQuA/{REVISION}/test.json'
PROTOCOL = 'diffa-aqua-rat-neutral-angry-bare-n50-v1'
COUNT = 50
CONDITIONS = ('neutral', 'angry')
PROMPT = ('Answer the question in the audio. End your response with a final answer line in the format:\n'
          'The answer is (<letter>).\nReplace <letter> with A, B, C, D, or E.')
METRICS = ('choice-accuracy', 'strict-final-line', 'format-compliant')
winners = base.winners


def speak(text):
    """Literal, answer-independent speech rendering; original notation remains in the bank."""
    text = text.replace('’', "'").replace('‘', "'").replace('–', '-')
    text = re.sub(r'Rs\.\s*(\d+(?:\.\d+)?)(?:/-)?', r'\1 rupees', text)
    text = re.sub(r'\$(\d+(?:\.\d+)?)', r'\1 dollars', text)
    text = re.sub(r'\b(\d+)C(\d+)\b', r'\1 choose \2', text)
    # Spell symbolic products/number names, without deciding their mathematical meaning.
    text = re.sub(r'\b(?:abc|cba|xy|wx)\b', lambda m: ' '.join(m[0]), text)
    text = re.sub(r'(?<=\d)(?=[abcwxy]\b)|(?<=\b[abcwxy])(?=\d)', ' ', text)
    symbols = {'%': ' percent ', '*': ' times ', '+': ' plus ', '/': ' slash ',
        '^': ' to the power of ', '<': ' less than ', '>': ' greater than ', '=': ' equals ',
        '(': ' open parenthesis, ', ')': ', close parenthesis ', '|': ' vertical bar ', '½': ' one half '}
    text = ''.join(symbols.get(c, c) for c in text)
    # Preserve lexical hyphens such as yo-yo; speak mathematical minus explicitly.
    text = re.sub(r'(?<=[A-Za-z0-9])-(?=[A-Za-z]{2,})', '\ue000', text)
    text = re.sub(r'(?<![A-Za-z])-|-(?![A-Za-z])', ' minus ', text).replace('\ue000', '-')
    return re.sub(r'\s+', ' ', text).strip()


def source_questions(data):
    blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if blob != SOURCE_BLOB:
        raise ValueError('AQuA source differs from the pinned authors\' Git blob.')
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if len(rows) != 254:
        raise ValueError('Expected all 254 AQuA-RAT test questions.')
    for row in rows:
        if (not isinstance(row['question'], str) or len(row['options']) != 5
                or any(not re.fullmatch(letter+r'\).+', option, re.S)
                       for letter, option in zip('ABCDE', row['options'])) or row['correct'] not in 'ABCDE'
                or len(row['correct']) != 1):
            raise ValueError('Unexpected source question/options/answer schema.')
    indexed = list(enumerate(rows)); random.Random(42).shuffle(indexed)
    questions = []
    for index, row in indexed[:COUNT]:
        spoken = speak(row['question'])+'\nOptions:\n'+'\n'.join(
            f'Option {letter}: {speak(option[2:])}' for letter, option in zip('ABCDE', row['options']))
        questions.append(dict(question_id=f'aqua_rat_test_{index:03d}', source_index=index,
            original_question=row['question'], original_options=row['options'],
            original_target=row['correct'], question=spoken, answer=row['correct']))
    return questions


def choice(response):
    """Last explicit answer declaration only; never mine incidental A-E option mentions."""
    clean = response.replace('*', '').replace('`', '').replace('_', '')
    matches = re.findall(r'^\s*The answer is\s+\(?([A-E])\)?[.!]?\s*$', clean, re.I | re.M)
    return matches[-1].upper() if matches else None


def score(response, target):
    lines = response.strip().splitlines()
    strict = re.fullmatch(r'The answer is \(([A-E])\)\.', lines[-1]) if lines else None
    prediction = choice(response)
    return dict(prediction=prediction, metrics={'choice-accuracy': prediction == target,
        'strict-final-line': bool(strict and strict[1] == target), 'format-compliant': bool(strict)})


def prepare(source, output):
    source.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    path = source/'aqua-test-source.jsonl'
    data = path.read_bytes() if path.exists() else download(SOURCE_URL)
    questions = source_questions(data)
    if not path.exists(): path.write_bytes(data)
    bank_file(source/'questions.jsonl', questions)
    selection = dict(protocol=PROTOCOL, name='aqua_rat', task='aqua_rat', model='diffa',
        prompt=PROMPT, prompts={c:PROMPT for c in CONDITIONS}, conditions=list(CONDITIONS),
        source_url=SOURCE_URL, source_revision=REVISION, source_git_blob=SOURCE_BLOB,
        source_sha256=hashlib.sha256(data).hexdigest(), seed=42, max_new_tokens=512,
        bank={q['question_id']:q for q in questions}, question_order=[q['question_id'] for q in questions],
        recipe=dict(base.evaluation.night.vanilla.RECIPE), voice=base.VOICE, azure_intensity=1.5,
        primary_metric='choice-accuracy',
        adaptation='Custom generated-answer audio protocol, not the AGIEval likelihood protocol. '
            'Fixed seed42 sample of50/254 authors\' test questions, no filtering. Both deliveries speak '
            'the same question and all five options. Literal symbol verbalization is saved beside the source: '
            'parentheses, powers, comparisons, arithmetic signs, currency and percent are spoken explicitly; '
            'slash and vertical bar are named literally. Suffix digits such as a2 are read as printed, not '
            'reinterpreted as exponents. Source wording/targets/typos are not repaired. '
            'Reference rationales never enter the question bank, audio or model prompt. '
            'No demonstrations, reasoning request, classifier filtering, activation collection or steering.')
    winners.freeze(output/'selection.json', selection)
    winners.atomic_text(output/'prompt.txt', PROMPT+'\n')
    return selection


def synthesize(source, prepare_only=False):
    command = [sys.executable, '-u', 'cloud_tts_smoke.py', '--provider', 'azure',
        '--azure-voice', base.VOICE, '--azure-intensity', '1.5', '--input-jsonl', str(source/'questions.jsonl'),
        '--num-examples', str(COUNT), '--styles', *CONDITIONS, '--output', str(source), '--resume']
    if prepare_only: command.append('--prepare-only')
    subprocess.run(command, check=True)


def plan_for(selection, source):
    records = source_records(source, selection['bank'], CONDITIONS)
    return dict(selection, records=[dict(records[q,c], condition=c,
        reference_answer=selection['bank'][q]['answer']) for q in selection['question_order'] for c in CONDITIONS])


def generate(plan, output, backend):
    done = base.raw_rows(output, plan, repair=True)
    checksum = winners.digest(output/'run_config.json')
    with (output/'deliveries.jsonl').open('a') as stream:
        for record in plan['records']:
            key = record['question_id'], record['condition']
            if key in done: continue
            print(f'AQuA [{len(done)+1}/{len(plan["records"])}] {key[0]}/{key[1]}', flush=True)
            started = time.monotonic()
            trace = backend.generate(record, plan, capture=False)
            row = dict(record, prompt=PROMPT, response=trace['response'], response_tokens=trace['response_tokens'],
                audio_encoding=trace['audio_encoding'], generation_seconds=time.monotonic()-started,
                model='diffa', strength=0., interventions=[], run_config_sha256=checksum, **plan['recipe'])
            stream.write(json.dumps(row)+'\n'); stream.flush(); done[key] = row
            print(f'  Saved tokens={row["response_tokens"]}; elapsed={row["generation_seconds"]:.1f}s', flush=True)


def report(output):
    plan = json.loads((output/'run_config.json').read_text())
    if plan['protocol'] != PROTOCOL or plan['prompt'] != PROMPT:
        raise ValueError('Frozen AQuA protocol differs.')
    rows = [dict(r, **score(r['response'], r['reference_answer'])) for r in base.raw_rows(output, plan).values()]
    groups = {c:{r['question_id']:r for r in rows if r['condition'] == c} for c in CONDITIONS}
    accuracy, lengths, comparisons = [], {}, []
    print(f'\nAQuA-RAT: {len(rows)}/{len(plan["records"])} answers; custom audio/generative scoring.', flush=True)
    for c, group in groups.items():
        if not group: continue
        tokens = [r['response_tokens'] for r in group.values()]
        lengths[c] = dict(n=len(group), mean_tokens=mean(tokens), median_tokens=median(tokens))
        print(f'{c}: N={len(group)}; mean/median tokens={mean(tokens):.1f}/{median(tokens):g}', flush=True)
        for metric in METRICS:
            correct = sum(r['metrics'][metric] for r in group.values())
            accuracy.append(dict(condition=c, metric=metric, n=len(group), correct=correct))
            print(f'  {metric}: {correct}/{len(group)} ({correct/len(group):.1%})', flush=True)
    ids = sorted(groups['neutral'].keys() & groups['angry'].keys())
    if ids:
        matched_lengths = {c:dict(mean_tokens=mean(groups[c][q]['response_tokens'] for q in ids),
            median_tokens=median(groups[c][q]['response_tokens'] for q in ids)) for c in CONDITIONS}
        print(f'Matched N={len(ids)}; neutral -> angry token means '
            f'{matched_lengths["neutral"]["mean_tokens"]:.1f} -> {matched_lengths["angry"]["mean_tokens"]:.1f}', flush=True)
        for metric in METRICS:
            stats = paired([(groups['neutral'][q]['metrics'][metric], groups['angry'][q]['metrics'][metric]) for q in ids])
            comparisons.append(dict(metric=metric, matched_lengths=matched_lengths, **stats))
            print(f'angry vs neutral / {metric}: N={len(ids)}; gain={stats["gain_pp"]:+.2f} pp; '
                f'p={stats["mcnemar_exact_p"]:.4g}; CI={stats["ci95_gain_pp"]}', flush=True)
    winners.atomic_text(output/'delivery-rescored.jsonl', ''.join(json.dumps(r)+'\n' for r in rows))
    winners.atomic_text(output/'delivery-summary.json', json.dumps(dict(protocol=PROTOCOL, accuracy=accuracy,
        lengths=lengths, comparisons=comparisons, adaptation=plan['adaptation'],
        scorer=dict(primary='Exact option letter against authors\' correct field; explicit last answer declaration.',
            strict='Exact final nonempty line: The answer is (A). through (E).',
            runner_sha256=winners.digest(__file__), source_url=SOURCE_URL,
            note='Local generated-output adaptation, not official AGIEval acc/acc_norm likelihood scoring.'),
        statistics='Exact paired McNemar p, paired bootstrap95% CI; unadjusted exploratory matched completed IDs.'), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'prepare', 'evaluate', 'summary'), default='run')
    parser.add_argument('--source', type=Path, default=Path('outputs/azure-aqua50'))
    parser.add_argument('--output', type=Path, default=Path('outputs/diffa-aqua50'))
    args = parser.parse_args()
    config = args.output/'run_config.json'
    if args.mode == 'summary':
        if config.exists(): report(args.output)
        else:
            clips = [p for p in args.source.glob('*.wav') if p.stat().st_size > 44]
            print(f'AQuA: audio {len(clips)}/100 clips; evaluation not prepared yet.', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'runner.lock').open('a') as own:
        try: fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('AQuA pilot already active; use --mode summary.')
        if args.mode in ('run', 'prepare'):
            selection = prepare(args.source, args.output)
            print('AQuA-RAT: 50 questions, neutral + angry, bare prompt; 100 clips / 100 answers.\n'+PROMPT, flush=True)
            synthesize(args.source, prepare_only=args.mode == 'prepare')
            if args.mode == 'prepare':
                print('Prepared only; no cloud synthesis or GPU inference.', flush=True); return
            plan = plan_for(selection, args.source)
            winners.freeze(config, plan)
        else:
            plan = json.loads(config.read_text())
            if plan['protocol'] != PROTOCOL or plan['prompt'] != PROMPT:
                raise ValueError('Frozen AQuA protocol differs.')
        done = base.raw_rows(args.output, plan)
        if len(done) < len(plan['records']):
            for path, expected in {r['audio_file']:r['audio_sha256'] for r in plan['records']}.items():
                if winners.digest(path) != expected: raise ValueError('Audio changed after preparation.')
            Path('outputs').mkdir(exist_ok=True)
            with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
                print('Waiting for shared GPU lock and idle GPU; audio generation complete.', flush=True)
                fcntl.flock(gpu, fcntl.LOCK_EX)
                while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                    time.sleep(30)
                backend = base.evaluation.night.DiffaBackend()
                from src.dataloader import build_chat_prompt
                from diffa_model_receipts import model_revisions
                native, count = base.evaluation.context_guard(backend.tokenizer, PROMPT, build_chat_prompt)
                winners.atomic_text(args.output/'native-chat-prompt.txt', native)
                winners.freeze(args.output/'runtime.json', dict(model_revisions=model_revisions(),
                    packages={k:version(k) for k in ('torch', 'transformers')},
                    model_config=backend.model.config.to_dict(), native_prompt_tokens=count,
                    runner_sha256=winners.digest(__file__)))
                generate(plan, args.output, backend)
        report(args.output)


if __name__ == '__main__': main()
