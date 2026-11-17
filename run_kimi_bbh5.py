"""Kimi BBH five: ALL 250 delivery pairs; own mean on 125, L17 x1 on other 125."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

import run_kimi_multiarith as shared
from run_kimi_gsm8k import prepare_audio
import run_reasoning_transfer as reasoning

PROTOCOL = 'kimi-bbh5-all250-collect125-test125-L17-answer-x1-v1'
STEERED = 'neutral-BBH5-L17-answer-a1'
SOURCE_SHA256 = 'd2df31394b903b7f0e085d42a69572a092987856eac076799017d70ef266fa8f'
SOURCE_URL = f'https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{reasoning.BBH_REVISION}/bbh/logical_deduction_five_objects.json'
PROMPT = reasoning.BBH_PROMPT
EXPLICIT_PROMPT = PROMPT.replace("Let's think step by step.",
    'Give concise step-by-step reasoning with only the necessary deductions.')


def select(args):
    source, output = args.source, args.output
    source.mkdir(parents=True, exist_ok=True)
    path = source/'logical_deduction_five_objects-source.json'
    if not path.exists():
        existing = next((p/path.name for p in args.audio_sources if (p/path.name).exists()), None)
        data = existing.read_bytes() if existing else reasoning.download(SOURCE_URL)
        if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
            raise ValueError('BBH-five source checksum differs.')
        path.write_bytes(data)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('Cached BBH-five source checksum differs.')
    questions = reasoning.bbh_questions(data, 'five')
    random.Random(42).shuffle(questions)
    train, test = questions[:125], questions[125:]
    reasoning.bank_file(source/'questions.jsonl', questions)
    explicit = getattr(args, 'explicit_reasoning', False)
    selection = dict(protocol=PROTOCOL+('-explicit-reasoning' if explicit else ''),
        prompt_variant='explicit-reasoning' if explicit else 'original',
        source_url=SOURCE_URL, source_sha256=SOURCE_SHA256,
        collection=train, test=test, prompt=EXPLICIT_PROMPT if explicit else PROMPT, seed=42, layer=17, group='answer', strength=1.,
        primary_metric='flexible-extract', primary_comparison='L17 x1 versus neutral on all 125 collection-unseen questions',
        rule='Full 250-question neutral/angry evaluation; seed42 125/125 split; collect all first-half pairs, '
             'steer only other half; no outcome/classifier/duration filter. L17 x1 fixed before Kimi BBH-five results.')
    shared.kimi.freeze(output/'selection.json', selection)
    return selection


def prepare(args, selection):
    plan = shared.prepare(args, selection, dataset='BBH-five', steered_condition=STEERED)
    return plan


def save_collection_references(plan, output):
    """The unmodified collection generations also supply the full-cohort references."""
    import torch
    done = {(r['question_id'], r['voice'], r['condition']) for r in shared.read_rows(output/'results.jsonl', repair=True)}
    with (output/'results.jsonl').open('a') as stream:
        for i, pair in enumerate(plan['collection']):
            cached = torch.load(output/f'pairs/{i:05d}.pt', map_location='cpu', weights_only=True)
            if cached['question_id'] != pair['question_id'] or cached['protocol'] != plan['protocol']:
                raise ValueError('Collection cache differs from frozen BBH-five split.')
            for style in ('neutral', 'angry'):
                key = pair['question_id'], pair['voice'], style
                if key in done:
                    continue
                if cached['audio_sha256'][style] != pair[style]['audio_sha256']:
                    raise ValueError('Collection recording changed.')
                row = dict(pair[style], condition=style, response=cached['responses'][style],
                    response_tokens=cached['response_tokens'][style], prompt=plan['prompt'], seed=42,
                    max_new_tokens=512, model=plan['model'], revision=plan['revision'], interventions=[],
                    vector_sha256=None, reused_from_collection=True)
                stream.write(json.dumps(row)+'\n'); stream.flush(); done.add(key)


def report(output):
    from score_bbh_harness import HarnessScorer
    from rescore_audio_steering import paired
    path = output/'run_config.json'
    if not path.exists():
        print('Kimi BBH five: preparation / waiting.'); return
    plan = json.loads(path.read_text()); rows = shared.read_rows(output/'results.jsonl')
    if not rows:
        print('Kimi BBH five: collection / waiting; no evaluation answers yet.'); return
    bank = {**plan['collection_bank'], **plan['bank']}
    for row in rows:
        if row['reference_answer'] != bank[row['question_id']]['answer'] or row['prompt'] != plan['prompt']:
            raise ValueError('Saved reference/prompt differs from frozen BBH-five protocol.')
        if row['condition'] == STEERED and row['question_id'] not in plan['bank']:
            raise ValueError('Steered collection question found; only held-out steering is allowed.')
    scorer = HarnessScorer(output/'harness/upstream', objects='five')
    docs = [dict(input=bank[r['question_id']]['original_input'], target=bank[r['question_id']]['original_target']) for r in rows]
    scores = scorer.score([r['response'] for r in rows], docs)
    # Diagnostic runs the SAME strict metric against every valid five-choice target.
    expanded = [r['response'] for r in rows for _ in 'ABCDE']
    alternatives = [dict(doc, target=f'({c})') for doc in docs for c in 'ABCDE']
    formats = scorer.score(expanded, alternatives)['strict-match']
    scored = [dict(row, scores={**{m: v[i]['correct'] for m, v in scores.items()},
        'strict-format-extractable': any(v['correct'] for v in formats[5*i:5*i+5])}) for i, row in enumerate(rows)]
    groups = {c: {} for c in ('neutral', 'angry', STEERED)}
    for row in scored:
        key = row['question_id'], row['voice']; group = groups[row['condition']]
        if key in group and group[key]['response'] != row['response']:
            raise ValueError('Conflicting saved answers.')
        group[key] = row
    accuracy, comparisons = [], []
    print(f'Kimi BBH five ({plan["selection"].get("prompt_variant", "original")}): '
          f'{len(rows)}/625 answers; full 250 delivery pairs, held-out 125 steering.')
    for cohort, conditions in [('all-250-delivery', ('neutral', 'angry')),
                               ('held-out-125-steering', ('neutral', 'angry', STEERED))]:
        keys = set.intersection(*(set(groups[c]) for c in conditions))
        if cohort.startswith('held-out'):
            keys = {k for k in keys if k[0] in plan['bank']}
        keys = sorted(keys); n = len(keys)
        print(f'\n{cohort}: {n} identical questions completed in all requested conditions.')
        if not n:
            continue
        for condition in conditions:
            counts = {m: sum(groups[condition][k]['scores'][m] for k in keys)
                      for m in ('flexible-extract', 'strict-match', 'strict-format-extractable')}
            lengths = [groups[condition][k]['response_tokens'] for k in keys]
            accuracy.append(dict(cohort=cohort, condition=condition, n=n, correct=counts,
                                 mean_tokens=statistics.mean(lengths), median_tokens=statistics.median(lengths)))
            print(f'{condition:32} N={n}: flexible={counts["flexible-extract"]}/{n} ({counts["flexible-extract"]/n:.1%}); '
                  f'strict={counts["strict-match"]}/{n} ({counts["strict-match"]/n:.1%}); '
                  f'format={counts["strict-format-extractable"]/n:.1%}; mean/median tokens={statistics.mean(lengths):.1f}/{statistics.median(lengths):.1f}')
        for metric in ('flexible-extract', 'strict-match', 'strict-format-extractable'):
            for condition in conditions[1:]:
                result = paired([(groups['neutral'][k]['scores'][metric], groups[condition][k]['scores'][metric]) for k in keys])
                comparisons.append(dict(cohort=cohort, metric=metric, condition=condition, **result))
                print(f'{metric} / {condition}: gain={result["gain_pp"]:+.2f} pp; W/L={result["wins"]}/{result["losses"]}; '
                      f'p={result["mcnemar_exact_p"]:.4f}; CI={result["ci95_gain_pp"]}')
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    (output/'summary.json').write_text(json.dumps(dict(protocol=plan['protocol'], prompt=plan['prompt'],
        prompt_variant=plan['selection'].get('prompt_variant', 'original'), primary_metric='flexible-extract',
        primary_comparison=plan['selection']['primary_comparison'], accuracy=accuracy, paired=comparisons,
        scorer=scorer.provenance, caveat='Installed upstream BBH-five scoring on audio-adapted generations. '
        'All-250 contains vector collection questions; steering comparison includes only the other 125. '
        'Same L17 x1 fixed beforehand; no outcome filtering. Secondary comparisons are unadjusted.'), indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['prepare', 'audio', 'evaluate', 'summary'], default='evaluate')
    p.add_argument('--source', type=Path, default=Path('outputs/azure-kimi-bbh-five250'))
    p.add_argument('--output', type=Path)
    p.add_argument('--explicit-reasoning', action='store_true', help='Explicit concise-reasoning prompt, with its own vector.')
    p.add_argument('--model-source', type=Path, default=Path('external/Kimi-Audio'))
    p.add_argument('--audio-sources', nargs='+', type=Path, default=[Path('outputs/azure-bbh-five250')])
    args = p.parse_args()
    if args.output is None:
        args.output = Path('outputs/kimi-bbh5-collect125-test125-L17'+('-explicit-reasoning' if args.explicit_reasoning else ''))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        report(args.output); return
    with (args.output/'queue.lock').open('w') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection = select(args)
        if args.mode == 'prepare':
            print('Frozen all 250 deliveries; collect 125 / steer other 125, L17 x1.'); return
        if args.mode == 'audio':
            prepare_audio(args, selection); prepare(args, selection); return
        plan = prepare(args, selection)
        done = {(r['question_id'], r['voice'], r['condition']) for r in shared.read_rows(args.output/'results.jsonl')}
        wanted = {(q['question_id'], q['voice'], c) for q in plan['collection']+plan['test'] for c in ('neutral', 'angry')}
        wanted |= {(q['question_id'], q['voice'], STEERED) for q in plan['test']}
        if wanted <= done:
            print('All 625 answers already saved; no model loading.'); return
        with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
            print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                print('Waiting for active GPU work.', flush=True); time.sleep(30)
            model = shared.kimi.load_model(args.model_source)
            vector = shared.collect(plan, args.output, model)
            save_collection_references(plan, args.output)
            shared.evaluate(plan, args.output, model, vector)


if __name__ == '__main__':
    main()
