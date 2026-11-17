"""Kimi MultiArith: fixed random 290/290 split, own mean, L17 ANSWER x1.

No recovery selection, layer search, base comparison or cross-model vector reuse.
Math prompt and upstream answer extraction are fixed before Kimi outcomes.
"""

import argparse
import fcntl
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import wave

import kimi_bbh_layer_diagnostic as kimi
import run_multiarith_mean as multiarith
from steer import read_rows

LAYER = 17
STEERED = 'neutral-MultiArith-L17-answer-a1'
PROTOCOL = 'kimi-multiarith-random290-L17-answer-x1-v1'
DEFAULT_SOURCE = Path('outputs/azure-multiarith580')
DEFAULT_OUTPUT = Path('outputs/kimi-multiarith290-L17')


def select(source, output):
    """Freeze questions and L17 before spending on TTS or observing Kimi answers."""
    source.mkdir(parents=True, exist_ok=True)
    original = source/'MultiArith-source.json'
    if not original.exists():
        data = multiarith.reasoning.download(multiarith.SOURCE_URL)
        multiarith.split_questions(data)
        original.write_bytes(data)
    train, test = multiarith.split_questions(original.read_bytes())
    multiarith.reasoning.bank_file(source/'questions.jsonl', train+test)
    selection = dict(protocol=PROTOCOL, model='moonshotai/Kimi-Audio-7B-Instruct',
        source_url=multiarith.SOURCE_URL, source_sha256=multiarith.SOURCE_SHA256,
        source_rows=600, unique_questions=580, seed=42,
        collection=train, test=test, prompt=multiarith.PROMPT,
        layer=LAYER, group='answer', strength=1., primary_metric='flexible-numeric',
        primary_comparison='neutral steered at fixed L17 x1 versus unsteered neutral on ALL 290 test questions',
        rule='Original deduplication and seed42 shuffle; first 290 collection and last 290 test. '
             'No classifier, correctness, response-length, or outcome filtering. '
             'Layer/group/strength fixed from the prior BBH experiment before any Kimi MultiArith outcomes.')
    kimi.freeze(output/'selection.json', selection)
    return selection


def audio(args):
    from run_reasoning_transfer import generate_audio
    selection = select(args.source, args.output)
    generate_audio(args.source, selection['collection']+selection['test'], ['neutral', 'angry'])
    return selection


def prepare(args, selection=None, dataset='MultiArith', steered_condition=STEERED):
    from evaluate_kimi_audio import MODEL, REVISION, SOURCE_REVISION, TOKENIZER_MODEL, TOKENIZER_REVISION
    selection = select(args.source, args.output) if selection is None else selection
    questions = selection['collection']+selection['test']
    bank = {q['question_id']: q for q in questions}
    groups = {}
    for row in read_rows(args.source/'plan.jsonl'):
        if row['provider'] != 'azure' or row['style'] not in ('neutral', 'angry'):
            continue
        q = bank[row['question_id']]
        if row['question'] != q['question'] or str(row['answer']).rsplit('####', 1)[-1].strip() != str(q['answer']):
            raise ValueError('MultiArith audio wording or gold differs from the frozen source.')
        path = (args.source/row['audio_path']).absolute()
        with wave.open(str(path)) as stream:
            seconds = stream.getnframes()/stream.getframerate()
        if seconds <= 0:
            raise ValueError(f'Empty recording: {path}')
        group = groups.setdefault(row['question_id'], {})
        if row['style'] in group:
            raise ValueError('Duplicate question/delivery in MultiArith plan.')
        group[row['style']] = dict(question_id=row['question_id'], voice=row['voice'], style=row['style'],
            audio_file=str(path), audio_sha256=kimi.digest(path), seconds=seconds, reference_answer=q['answer'])
    if set(groups) != set(bank) or any(set(g) != {'neutral', 'angry'} for g in groups.values()):
        raise ValueError('Need complete neutral/angry audio for all 580 unique questions; run --mode audio first.')
    if any(g['neutral']['voice'] != g['angry']['voice'] for g in groups.values()):
        raise ValueError('Neutral/angry voices differ.')
    def pair(q):
        return dict(question_id=q['question_id'], voice=groups[q['question_id']]['neutral']['voice'], **groups[q['question_id']])
    train = [pair(q) for q in selection['collection']]
    test = [pair(q) for q in selection['test']]
    if {p['question_id'] for p in train} & {p['question_id'] for p in test}:
        raise ValueError('Collection and test questions overlap.')
    plan = dict(protocol=selection.get('protocol', PROTOCOL), selection=selection, prompt=selection.get('prompt', multiarith.PROMPT), seed=42,
        model=MODEL, revision=REVISION, source_revision=SOURCE_REVISION,
        tokenizer_model=TOKENIZER_MODEL, tokenizer_revision=TOKENIZER_REVISION,
        generation=kimi.PARAMS, layer=LAYER, group='answer', strength=1.,
        collection=train, test=test, collection_bank={q['question_id']: q for q in selection['collection']},
        bank={q['question_id']: q for q in selection['test']},
        vector=str((args.output/'style-vectors.pt').absolute()),
        vector_definition='Per-question mean angry minus neutral L17 block-output answer-prediction state, '
                          f'then equal mean across all {len(train)} collection pairs. Native greedy histories; raw magnitude retained.',
        audio_encoding='Full recordings through native Kimi encoders; no DIFFA adapter or 30-second eligibility filter.',
        scoring='Installed upstream GSM8K extraction/metric on MultiArith numeric references; additional numeric equivalence. '
                'Audio-adapted inference, not a dedicated official MultiArith harness task.')
    if dataset != 'MultiArith':
        plan.update(dataset=dataset, steered_condition=steered_condition,
            scoring=('Installed upstream BBH-five filters and exact-match metric on original BBH-five targets. '
                     'Audio-adapted inference, not full official text-harness inference.' if dataset == 'BBH-five' else
                     'Installed upstream GSM8K extraction/metric on GSM8K references; additional numeric equivalence. '
                     'Audio-adapted inference, not full official text-harness inference.'))
    kimi.freeze(args.output/'run_config.json', plan)
    return plan


def text_tokens(model, response):
    return len(model.prompt_manager.text_tokenizer.encode(response, bos=False, eos=False))


def collect(plan, output, model):
    import torch
    protocol = plan.get('protocol', PROTOCOL)
    folder = output/'pairs'; folder.mkdir(exist_ok=True)
    for i, pair in enumerate(plan['collection']):
        path = folder/f'{i:05d}.pt'
        if path.exists():
            cached = torch.load(path, map_location='cpu', weights_only=True)
            if (cached['protocol'] != protocol or cached['question_id'] != pair['question_id']
                    or cached['audio_sha256'] != {s: pair[s]['audio_sha256'] for s in ('neutral', 'angry')}):
                raise ValueError('Collection cache differs from the frozen 290-pair split.')
            continue
        print(f'Collect [{i+1}/{len(plan["collection"])}] {pair["question_id"]}: neutral + angry, L17', flush=True)
        traces = {}
        for style in ('neutral', 'angry'):
            if kimi.digest(pair[style]['audio_file']) != pair[style]['audio_sha256']:
                raise ValueError('Collection audio changed.')
            traces[style] = kimi.trace_generation(model, pair[style], plan['prompt'])
        # Do not store 28-layer traces or large encoded-audio tensors: only L17 summaries.
        saved = dict(protocol=protocol, question_id=pair['question_id'], voice=pair['voice'],
            audio_sha256={s: pair[s]['audio_sha256'] for s in traces},
            responses={s: traces[s]['response'] for s in traces},
            response_tokens={s: text_tokens(model, traces[s]['response']) for s in traces},
            forwards={s: len(traces[s]['calls']) for s in traces},
            answer_difference=traces['angry']['means'][LAYER]-traces['neutral']['means'][LAYER],
            audio_difference=traces['angry']['audio_means'][LAYER]-traces['neutral']['audio_means'][LAYER])
        kimi.atomic_save(path, saved)
    cached = [torch.load(folder/f'{i:05d}.pt', map_location='cpu', weights_only=True) for i in range(len(plan['collection']))]
    vector = dict(protocol=protocol, model=plan['model'], revision=plan['revision'],
        layer=LAYER, strength=1., prompt=plan['prompt'],
        collection_question_ids=[p['question_id'] for p in plan['collection']],
        answer_direction=torch.stack([r['answer_difference'] for r in cached]).mean(0),
        audio_direction=torch.stack([r['audio_difference'] for r in cached]).mean(0),
        applied_group='answer', audio_note='L17 audio difference saved for reference; no audio steering in this fixed answer-only test.')
    vector_path = Path(plan['vector'])
    if vector_path.exists():
        stored = torch.load(vector_path, map_location='cpu', weights_only=True)
        if (stored['protocol'] != protocol or stored['collection_question_ids'] != vector['collection_question_ids']
                or stored['prompt'] != vector['prompt']
                or any(not torch.equal(stored[k], vector[k]) for k in ('answer_direction', 'audio_direction'))):
            raise ValueError('Saved vector differs from the frozen collection; use a new output folder.')
        return stored
    kimi.atomic_save(vector_path, vector)
    return vector


def evaluate(plan, output, model, vector):
    steered = plan.get('steered_condition', STEERED)
    if vector['collection_question_ids'] != [p['question_id'] for p in plan['collection']]:
        raise ValueError('Vector collection IDs differ from the frozen split.')
    done = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(output/'results.jsonl', repair=True)}
    tasks = [(p, c) for p in plan['test'] for c in ('neutral', 'angry', steered)
             if (p['question_id'], p['voice'], c) not in done]
    with (output/'results.jsonl').open('a') as stream:
        for i, (pair, condition) in enumerate(tasks, 1):
            row = pair['angry' if condition == 'angry' else 'neutral']
            if kimi.digest(row['audio_file']) != row['audio_sha256']:
                raise ValueError('Test audio changed.')
            print(f'Evaluate [{i}/{len(tasks)}] {pair["question_id"]} {condition}', flush=True)
            started = time.monotonic()
            if condition == steered:
                response = kimi.generate_steered(model, row, plan['prompt'], LAYER, 'answer', vector['answer_direction'])
            else:
                response = kimi.generate(model, row, plan['prompt'])
            saved = dict(row, condition=condition, response=response, prompt=plan['prompt'], seed=42,
                max_new_tokens=512, model=plan['model'], revision=plan['revision'],
                vector_sha256=kimi.digest(Path(plan['vector'])) if condition == steered else None,
                response_tokens=text_tokens(model, response), generation_seconds=time.monotonic()-started,
                interventions=[dict(layer=LAYER, group='answer', strength=1.)] if condition == steered else [])
            stream.write(json.dumps(saved)+'\n'); stream.flush()
            print(response, flush=True)


def report(output):
    from rescore_audio_steering import HarnessTaskScorer, numeric_equal, paired
    path = output/'run_config.json'
    if not path.exists():
        print('Kimi MultiArith: audio preparation / waiting; no model evaluation configuration yet.'); return
    plan = json.loads(path.read_text())
    steered = plan.get('steered_condition', STEERED)
    rows = read_rows(output/'results.jsonl')
    if not rows:
        print('Kimi MultiArith: collection / waiting; no test answers saved yet.'); return
    allowed = {p['question_id'] for p in plan['test']}
    if any(r['question_id'] not in allowed for r in rows):
        raise ValueError('Collection question found in test answers.')
    scorer = HarnessTaskScorer('gsm8k', output/'harness/upstream')
    docs = [dict(question=plan['bank'][r['question_id']]['question'], answer=plan['bank'][r['question_id']]['answer']) for r in rows]
    for r, doc in zip(rows, docs):
        if r['reference_answer'] != doc['answer'] or r['prompt'] != plan['prompt']:
            raise ValueError('Saved question reference or prompt changed.')
    values = scorer.score([r['response'] for r in rows], docs)
    scored = []
    for i, row in enumerate(rows):
        metrics = {m: v[i]['correct'] for m, v in values.items()}
        metrics['flexible-numeric'] = numeric_equal(values['flexible-extract'][i]['prediction'], row['reference_answer'])
        metrics['format-compliant'] = values['strict-match'][i]['prediction'] != '[invalid]'
        scored.append(dict(row, scores=metrics, predictions={m: v[i]['prediction'] for m, v in values.items()}))
    groups = {c: {(r['question_id'], r['voice']): r for r in scored if r['condition'] == c}
              for c in ('neutral', 'angry', steered)}
    common = sorted(set.intersection(*(set(g) for g in groups.values())))
    print(f'\nKimi {plan.get("dataset", "MultiArith")}: {len(rows)}/{len(plan["test"])*3} test answers; '
          f'{len(common)} questions completed in ALL THREE conditions. L17 answer x1 fixed before outcomes.')
    print('Condition                              N  Numeric   Flexible  Strict  Format  Mean tokens  Median tokens')
    accuracy, comparisons = [], []
    for condition, group in groups.items():
        if not common:
            continue
        counts = {m: sum(group[k]['scores'][m] for k in common) for m in ('flexible-numeric', 'flexible-extract', 'strict-match', 'format-compliant')}
        lengths = [group[k]['response_tokens'] for k in common]
        n = len(common)
        accuracy.append(dict(condition=condition, n=n, correct=counts,
            mean_tokens=statistics.mean(lengths), median_tokens=statistics.median(lengths)))
        print(f'{condition:37} {n:3d} {counts["flexible-numeric"]/n:8.1%} {counts["flexible-extract"]/n:8.1%}'
              f' {counts["strict-match"]/n:7.1%} {counts["format-compliant"]/n:7.1%}'
              f' {statistics.mean(lengths):11.1f} {statistics.median(lengths):14.1f}')
    for metric in ('flexible-numeric', 'flexible-extract', 'strict-match', 'format-compliant'):
        for condition in ('angry', steered):
            keys = sorted(groups['neutral'].keys() & groups[condition].keys())
            if not keys:
                continue
            result = paired([(groups['neutral'][k]['scores'][metric], groups[condition][k]['scores'][metric]) for k in keys])
            comparisons.append(dict(metric=metric, condition=condition, **result))
            print(f'{metric} / {condition}: N={len(keys)}; gain={result["gain_pp"]:+.2f} pp; '
                  f'W/L={result["wins"]}/{result["losses"]}; p={result["mcnemar_exact_p"]:.4f}; CI={result["ci95_gain_pp"]}')
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    (output/'summary.json').write_text(json.dumps(dict(protocol=plan.get('protocol', PROTOCOL), primary_metric='flexible-numeric',
        primary_comparison=plan['selection']['primary_comparison'], matched_all_three=len(common), accuracy=accuracy,
        paired=comparisons, scorer=scorer.provenance,
        note=plan['scoring']+f' Secondary comparisons are exploratory/unadjusted. All {len(plan["test"])} test questions are disjoint '
             'from vector collection and were not selected by Kimi outcomes.'), indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['all', 'prepare', 'audio', 'evaluate', 'summary'], default='all')
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--model-source', type=Path, default=Path('external/Kimi-Audio'))
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        report(args.output); return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == 'prepare':
            select(args.source, args.output); print('Frozen 290/290 split and L17 answer x1; no TTS or inference.'); return
        if args.mode in ('all', 'audio'):
            audio(args)
        if args.mode == 'audio':
            prepare(args); return
        plan = prepare(args)
        complete = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(args.output/'results.jsonl')}
        test_done = all((p['question_id'], p['voice'], c) in complete
                        for p in plan['test'] for c in ('neutral', 'angry', STEERED))
        if test_done and Path(plan['vector']).exists():
            print('All test answers already saved; no model loading required.', flush=True)
            return
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            model = kimi.load_model(args.model_source)
            vector = collect(plan, args.output, model)
            evaluate(plan, args.output, model, vector)


if __name__ == '__main__':
    main()
