"""50 BBH arithmetic questions: neutral/angry × bare/supply reasoning, no steering."""
import argparse
import ast
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
from rescore_audio_steering import HarnessTaskScorer, download, paired
from run_reasoning_transfer import bank_file, source_records

TASK = 'multistep_arithmetic_two'
SCORER = 'bbh_cot_zeroshot_'+TASK
REVISION = '9ee07bd481feebf959a6b59d61ea57bdcf30964d'
SOURCE_BLOB = '5d961d60f818c2df15342f4d20a3f07cf60caba1'
SOURCE_URL = f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{REVISION}/bbh/{TASK}.json'
PROTOCOL = 'diffa-bbh-arithmetic-audio-four-conditions-n50-v2'
COUNT = 50
STYLES = ('neutral', 'angry')
CONDITIONS = ('neutral', 'angry', 'neutral-reasoning', 'angry-reasoning')
BARE_PROMPT = base.BARE_PROMPT
REASONING_SENTENCE = 'Supply reasoning before the final answer.'
REASONING_PROMPT = BARE_PROMPT.replace('audio. ', 'audio. '+REASONING_SENTENCE+' ', 1)
PROMPTS = {c: REASONING_PROMPT if c.endswith('-reasoning') else BARE_PROMPT for c in CONDITIONS}
winners = base.winners


def expression_value(expression):
    """Evaluate only the integer +, -, * grammar present in this pinned source."""
    text = expression.strip()
    if not text.endswith('='):
        raise ValueError('Expected a trailing equals sign.')
    try:
        tree = ast.parse(text[:-1].strip(), mode='eval')
    except SyntaxError as exc:
        raise ValueError('Invalid arithmetic expression.') from exc

    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -value(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = value(node.left), value(node.right)
            if isinstance(node.op, ast.Add): return left+right
            if isinstance(node.op, ast.Sub): return left-right
            if isinstance(node.op, ast.Mult): return left*right
        raise ValueError('Unsupported arithmetic syntax; refusing to alter the source silently.')
    return value(tree.body)


def spoken_question(expression):
    expression_value(expression)
    text = expression.strip()[:-1].strip()
    tokens = re.findall(r'\d+|[()+*\-]', text)
    if ''.join(tokens) != re.sub(r'\s+', '', text):
        raise ValueError('Unrecognized symbol in arithmetic expression.')
    words, operand = [], True
    numbers = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
    for token in tokens:
        if token.isdigit():
            if int(token) > 9:
                raise ValueError('Pinned source is expected to use single-digit operands.')
            words.append(numbers[int(token)]); operand = False
        elif token == '(':
            words.append('open parenthesis,'); operand = True
        elif token == ')':
            words.append(', close parenthesis,'); operand = False
        else:
            words.append('negative' if token == '-' and operand else {'+':'plus', '-':'minus', '*':'times'}[token])
            operand = True
    spoken = re.sub(r'(?:,\s*)+', ', ', ' '.join(words).replace(' ,', ',')).rstrip(', ')
    return 'What is the value of: '+spoken+'?'


def source_questions(data):
    digest = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if digest != SOURCE_BLOB:
        raise ValueError('Arithmetic source differs from the pinned authors\' Git blob.')
    examples = json.loads(data)['examples']
    if len(examples) != 250:
        raise ValueError('Expected 250 official BBH arithmetic questions.')
    # Check every source target and transcription before selecting, without filtering.
    questions = []
    for i, x in enumerate(examples):
        spoken = spoken_question(x['input'])
        if str(expression_value(x['input'])) != x['target']:
            raise ValueError(f'Source arithmetic target mismatch at index {i}.')
        questions.append(dict(question_id=f'bbh_arithmetic_{i:03d}', source_index=i,
            original_input=x['input'], original_target=x['target'], question=spoken, answer=x['target']))
    random.Random(42).shuffle(questions)
    return questions[:COUNT]


def prepare(source, output):
    source.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    path = source/'bbh-source.json'
    data = path.read_bytes() if path.exists() else download(SOURCE_URL)
    questions = source_questions(data)
    if not path.exists(): path.write_bytes(data)
    bank_file(source/'questions.jsonl', questions)
    selection = dict(protocol=PROTOCOL, name=TASK, model='diffa', task=TASK,
        label='BBH arithmetic audio pilot', prompt=BARE_PROMPT, prompts=PROMPTS,
        source_url=SOURCE_URL, source_git_blob=SOURCE_BLOB, source_sha256=hashlib.sha256(data).hexdigest(),
        bank={q['question_id']:q for q in questions}, question_order=[q['question_id'] for q in questions],
        styles=list(STYLES), conditions=list(CONDITIONS), voice=base.VOICE, azure_intensity=1.5,
        recipe=dict(base.evaluation.night.vanilla.RECIPE), seed=42, max_new_tokens=512,
        primary_metric='flexible-extract',
        adaptation='Custom spoken adaptation of BBH multistep_arithmetic_two: fixed seed42 sample of50/250, '
            'no filtering. Every parenthesis spoken explicitly; unary minus spoken as negative, subtraction as minus; '
            'single-digit operands spoken as English words, addition as plus, multiplication as times. '
            'The same spoken question in Jenny neutral and angry; each WAV reused for its bare and reasoning '
            'conditions. Supply reasoning sentence in both reasoning conditions. No examples, classification, '
            'fitting or steering. Upstream saved-answer filters and exact-match scoring, not an official generation prompt.')
    winners.freeze(output/'selection.json', selection)
    winners.atomic_text(output/'prompt-bare.txt', BARE_PROMPT+'\n')
    winners.atomic_text(output/'prompt-reasoning.txt', REASONING_PROMPT+'\n')
    return selection


def synthesize(source, prepare_only=False):
    command = [sys.executable, '-u', 'cloud_tts_smoke.py', '--provider', 'azure',
        '--azure-voice', base.VOICE, '--azure-intensity', '1.5',
        '--input-jsonl', str(source/'questions.jsonl'), '--num-examples', str(COUNT),
        '--styles', *STYLES, '--output', str(source), '--resume']
    if prepare_only: command.append('--prepare-only')
    subprocess.run(command, check=True)


def plan_for(selection, source):
    records = source_records(source, selection['bank'], STYLES)
    return dict(selection, records=[dict(records[q, c.removesuffix('-reasoning')],
        condition=c, reference_answer=selection['bank'][q]['answer'])
        for q in selection['question_order'] for c in CONDITIONS])


def scoring_preflight(output):
    scorer = HarnessTaskScorer(SCORER, output/'scoring')
    targets = ['24', '-50', '-11520', '0']
    values = scorer.score(['The answer is '+n+'.' for n in targets],
        [dict(input='Format check only', target=n) for n in targets])
    if not all(v['correct'] for group in values.values() for v in group):
        raise ValueError('Final-answer format does not pass upstream arithmetic filters.')
    return scorer


def generate(plan, output, backend):
    done = base.raw_rows(output, plan, repair=True)
    checksum = winners.digest(output/'run_config.json')
    with (output/'deliveries.jsonl').open('a') as stream:
        for record in plan['records']:
            key = record['question_id'], record['condition']
            if key in done: continue
            prompt = plan['prompts'][record['condition']]
            print(f'Arithmetic [{len(done)+1}/{len(plan["records"])}] {key[0]}/{key[1]}', flush=True)
            trace = backend.generate(record, dict(plan, prompt=prompt), capture=False)
            row = dict(record, prompt=prompt, response=trace['response'], response_tokens=trace['response_tokens'],
                audio_encoding=trace['audio_encoding'], model='diffa', strength=0., interventions=[],
                run_config_sha256=checksum, **plan['recipe'])
            stream.write(json.dumps(row)+'\n'); stream.flush(); done[key] = row


def report(output):
    plan = json.loads((output/'run_config.json').read_text())
    scorer = HarnessTaskScorer(SCORER, output/'scoring')
    rows = list(base.raw_rows(output, plan).values())
    docs = [dict(input=plan['bank'][r['question_id']]['original_input'],
        target=plan['bank'][r['question_id']]['original_target']) for r in rows]
    values = scorer.score([r['response'] for r in rows], docs)
    scored = [dict(r, metrics={m:v[i]['correct'] for m,v in values.items()},
        predictions={m:v[i]['prediction'] for m,v in values.items()}) for i,r in enumerate(rows)]
    groups = {c:{r['question_id']:r for r in scored if r['condition']==c} for c in CONDITIONS}
    print(f'\nBBH Arithmetic: {len(rows)}/{len(plan["records"])} answers; 50 questions, four conditions.', flush=True)
    accuracy, lengths, comparisons = [], {}, []
    for c, group in groups.items():
        if not group: continue
        tokens = [r['response_tokens'] for r in group.values()]
        lengths[c] = dict(n=len(group), mean_tokens=mean(tokens), median_tokens=median(tokens))
        print(f'{c}: N={len(group)}; mean/median tokens={mean(tokens):.1f}/{median(tokens):g}', flush=True)
        for metric in values:
            correct = sum(r['metrics'][metric] for r in group.values())
            accuracy.append(dict(condition=c, metric=metric, n=len(group), correct=correct))
            print(f'  {metric}: {correct}/{len(group)} ({correct/len(group):.1%})', flush=True)
    contrasts = [('neutral', c) for c in CONDITIONS if c != 'neutral']
    contrasts.append(('neutral-reasoning', 'angry-reasoning'))
    for reference, test in contrasts:
        ids = sorted(groups[reference].keys() & groups[test].keys())
        if not ids: continue
        matched_lengths = {c:dict(mean_tokens=mean(groups[c][q]['response_tokens'] for q in ids),
            median_tokens=median(groups[c][q]['response_tokens'] for q in ids)) for c in (reference, test)}
        label = 'neutral-bare' if reference == 'neutral' else reference
        print(f'{test} vs {label}: matched token means '
            f'{matched_lengths[reference]["mean_tokens"]:.1f} -> {matched_lengths[test]["mean_tokens"]:.1f}', flush=True)
        for metric in values:
            stats = paired([(groups[reference][q]['metrics'][metric], groups[test][q]['metrics'][metric]) for q in ids])
            comparisons.append(dict(reference=reference, test=test, metric=metric, matched_lengths=matched_lengths, **stats))
            print(f'  {metric}: N={len(ids)}; gain={stats["gain_pp"]:+.2f} pp; '
                f'p={stats["mcnemar_exact_p"]:.4g}; CI={stats["ci95_gain_pp"]}', flush=True)
    winners.atomic_text(output/'delivery-rescored.jsonl', ''.join(json.dumps(r)+'\n' for r in scored))
    winners.atomic_text(output/'delivery-summary.json', json.dumps(dict(protocol=PROTOCOL, accuracy=accuracy,
        lengths=lengths, comparisons=comparisons, scorer=scorer.provenance, adaptation=plan['adaptation'],
        statistics='Exact paired McNemar p, paired bootstrap95% CI; unadjusted, matched completed IDs.'), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'prepare', 'evaluate', 'summary'), default='run')
    parser.add_argument('--source', type=Path, default=Path('outputs/azure-bbh-arithmetic50'))
    parser.add_argument('--output', type=Path, default=Path('outputs/diffa-bbh-arithmetic50'))
    args = parser.parse_args()
    config = args.output/'run_config.json'
    if args.mode == 'summary':
        if config.exists(): report(args.output)
        else:
            clips = [p for p in args.source.glob('*.wav') if p.stat().st_size > 44]
            print(f'BBH Arithmetic: audio {len(clips)}/100 clips; evaluation not prepared yet.', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'runner.lock').open('a') as own:
        try: fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Arithmetic pilot already active; use --mode summary.')
        scoring_preflight(args.output)
        if args.mode in ('run', 'prepare'):
            selection = prepare(args.source, args.output)
            print('BBH Arithmetic: 50 questions, neutral/angry x bare/supply reasoning; 100 clips, 200 answers. No steering.', flush=True)
            print('Bare prompt (neutral + angry):\n'+BARE_PROMPT+'\nReasoning prompt (neutral + angry):\n'+REASONING_PROMPT, flush=True)
            first = selection['bank'][selection['question_order'][0]]
            print('First source expression: '+first['original_input']+'\nSpoken: '+first['question'], flush=True)
            synthesize(args.source, prepare_only=args.mode == 'prepare')
            if args.mode == 'prepare':
                print('Prepared only; no cloud synthesis or GPU inference.', flush=True); return
            plan = plan_for(selection, args.source)
            winners.freeze(config, plan)
        else:
            plan = json.loads(config.read_text())
            if plan['protocol'] != PROTOCOL or plan['prompts'] != PROMPTS:
                raise ValueError('Frozen arithmetic protocol differs.')
        done = base.raw_rows(args.output, plan)
        if len(done) < len(plan['records']):
            for path, expected in {r['audio_file']:r['audio_sha256'] for r in plan['records']}.items():
                if winners.digest(path) != expected: raise ValueError('Audio changed after preparation.')
            with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
                print('Waiting for shared GPU lock and idle GPU; audio generation complete.', flush=True)
                fcntl.flock(gpu, fcntl.LOCK_EX)
                while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                    time.sleep(30)
                backend = base.evaluation.night.DiffaBackend()
                from src.dataloader import build_chat_prompt
                from diffa_model_receipts import model_revisions
                contexts = {}
                for label, prompt in (('bare', BARE_PROMPT), ('reasoning', REASONING_PROMPT)):
                    text, count = base.evaluation.context_guard(backend.tokenizer, prompt, build_chat_prompt)
                    winners.atomic_text(args.output/f'native-chat-prompt-{label}.txt', text); contexts[label] = count
                winners.freeze(args.output/'runtime.json', dict(model_revisions=model_revisions(),
                    packages={k:version(k) for k in ('torch', 'transformers')},
                    model_config=backend.model.config.to_dict(), native_prompt_tokens=contexts,
                    runner_sha256=winners.digest(__file__)))
                generate(plan, args.output, backend)
        report(args.output)


if __name__ == '__main__':
    main()
