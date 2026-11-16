"""All original 250 Kimi BBH-3 pairs; OLD prompt; L16/L17/L18 answer x1 separately.

Reuses the original 125-pair vectors and compatible completed diagnostic answers.
All 250 and collection-seen/unseen results are reported separately.
"""

import argparse
import fcntl
import json
from pathlib import Path
import statistics

import kimi_bbh_layer_diagnostic as diagnostic
import kimi_bbh_regressions as regression
from run_bbh_logical200 import PROMPT as OLD_PROMPT
from steer import read_rows

TARGETS = tuple(dict(layer=i, group='answer', strength=1.,
                     condition=diagnostic.condition_name(i, 'answer')) for i in (16, 17, 18))
DEFAULT_OUTPUT = Path('outputs/kimi-bbh3-full-L16-L17-L18')


def prepare(args):
    source = json.loads((args.recovery/'run_config.json').read_text())
    if source['protocol'] != diagnostic.PROTOCOL:
        raise ValueError('Expected the corrected audio+answer vector collection.')
    benchmark = Path(source['benchmark'])
    original = json.loads((benchmark/'run_config.json').read_text())
    if diagnostic.digest(benchmark/'run_config.json') != source['source_config_sha256']:
        raise ValueError('Original benchmark configuration changed since collection.')
    if original['prompt'] != source['prompt'] or source['prompt'] != OLD_PROMPT:
        raise ValueError('Full steering must use the OLD original prompt, not the explicit reasoning prompt.')
    if source['generation'] != diagnostic.PARAMS:
        raise ValueError('Vector generation recipe differs from the pinned old-prompt Kimi recipe.')
    pairs = diagnostic.matched_pairs(read_rows(benchmark/'rescored.jsonl'))
    if {p['question_id'] for p in pairs} != set(original['bank']):
        raise ValueError('Finish the original neutral/angry benchmark before full steering.')
    collection_ids = sorted(p['question_id'] for p in source['collection_pairs'])
    for pair in pairs:
        pair['in_vector_collection'] = pair['question_id'] in collection_ids
        for style in ('neutral', 'angry'):
            row = pair[style]
            if (row['prompt'] != source['prompt'] or row['seed'] != 42 or row['max_new_tokens'] != 512
                    or row['model'] != source['instruct_model'] or row['revision'] != source['instruct_revision']):
                raise ValueError('Original saved response settings differ from the collection recipe.')
            if diagnostic.digest(row['audio_file']) != row['audio_sha256']:
                raise ValueError(f'Original audio missing or changed: {row["audio_file"]}')
    vector = args.recovery/'style-vectors.pt'
    plan = dict(protocol='kimi-bbh3-full-old-prompt-L16-L17-L18-v1',
        recovery_directory=str(args.recovery.absolute()), benchmark=str(benchmark),
        source_config_sha256=source['source_config_sha256'], bank=original['bank'], pairs=pairs,
        collection_ids=collection_ids, overlap_with_collection=collection_ids,
        prompt=source['prompt'], seed=42, generation=source['generation'],
        instruct_model=source['instruct_model'], instruct_revision=source['instruct_revision'],
        source_revision=source['source_revision'], targets=list(TARGETS),
        vector_source=str(vector.absolute()), vector_sha256=diagnostic.digest(vector),
        evaluation_label='full-cohort OLD-prompt steering',
        selection='ALL original matched questions, no accuracy/classifier filtering. '
                  'L16/L17/L18 answer x1 chosen after recovery and regression diagnostics; exploratory.',
        caveat='All 250 includes the 125 vector-collection questions. Collection-unseen includes '
               'recovery questions used for layer selection and inspected regressions; it is not untouched confirmatory validation.')
    diagnostic.freeze(args.output/'run_config.json', plan)
    return plan


def reusable(row, task, target, plan):
    expected = task['neutral']
    for name in ('question_id', 'voice', 'audio_sha256', 'reference_answer'):
        if row.get(name) != expected[name]:
            return False
    if (row.get('prompt') != plan['prompt'] or row.get('seed') != 42 or row.get('max_new_tokens') != 512
            or row.get('model') != plan['instruct_model'] or row.get('revision') != plan['instruct_revision']):
        return False
    interventions = row.get('interventions', [])
    return len(interventions) == 1 and all(interventions[0].get(k) == target[k] for k in ('layer', 'group', 'strength'))


def reuse_answers(plan, output, sources):
    """Copy only exact old-prompt/audio/vector-compatible answers; never edit sources."""
    tasks = {(p['question_id'], p['voice']): p for p in plan['pairs']}
    targets = {t['condition']: t for t in plan['targets']}
    destination = output/'results.jsonl'
    existing = {(r['question_id'], r['voice'], r['condition']): r for r in read_rows(destination, repair=True)}
    copied = 0
    with destination.open('a') as stream:
        for source in sources:
            if not (source/'run_config.json').exists():
                continue
            config = json.loads((source/'run_config.json').read_text())
            if (config['prompt'] != plan['prompt'] or config['generation'] != plan['generation']
                    or config['instruct_model'] != plan['instruct_model']
                    or config['instruct_revision'] != plan['instruct_revision']):
                raise ValueError(f'Saved steering recipe differs: {source}')
            vector = Path(config.get('vector_source', source/'style-vectors.pt'))
            if diagnostic.digest(vector) != plan['vector_sha256']:
                raise ValueError(f'Saved answers use a different vector file: {source}')
            for row in read_rows(source/'results.jsonl'):
                target = targets.get(row['condition'])
                task = tasks.get((row['question_id'], row['voice']))
                if target is None or task is None:
                    continue
                if not reusable(row, task, target, plan):
                    raise ValueError(f'Incompatible saved steering answer: {source}, {row["question_id"]}, {row["condition"]}')
                key = row['question_id'], row['voice'], row['condition']
                if key in existing:
                    if existing[key]['response'] != row['response']:
                        raise ValueError(f'Conflicting saved steering responses: {key}')
                    continue
                saved = dict(row, in_vector_collection=task['in_vector_collection'], reused_from=str(source))
                stream.write(json.dumps(saved)+'\n'); stream.flush()
                existing[key] = saved; copied += 1
    print(f'Reused {copied} completed old-prompt steering answers. Sources retained.', flush=True)


def report(output):
    from score_bbh_harness import HarnessScorer
    from evaluate_bbh_audio_models import strict_format_bits
    from rescore_audio_steering import paired
    plan = json.loads((output/'run_config.json').read_text())
    rows = read_rows(output/'results.jsonl')
    if not rows:
        print('No full-cohort answers saved yet.'); return
    scorer = HarnessScorer(output/'harness/upstream')
    docs = [dict(input=plan['bank'][r['question_id']]['original_input'],
                 target=plan['bank'][r['question_id']]['original_target']) for r in rows]
    values = scorer.score([r['response'] for r in rows], docs)
    formats = strict_format_bits(scorer, [r['response'] for r in rows], docs)
    scored = [dict(r, harness={m: v[i] for m, v in values.items()},
                   **{'strict-format-extractable': formats[i]}) for i, r in enumerate(rows)]
    summaries, comparisons = [], []
    for subset in ('all', 'collection-unseen', 'collection-seen'):
        chosen = [r for r in scored if subset == 'all'
                  or bool(r['in_vector_collection']) == (subset == 'collection-seen')]
        print(f'\n{subset} — OLD prompt; original mean vectors; L16/L17/L18 answer x1 independently.')
        print('Condition                        N  Flexible  Strict  Valid format  Mean words  Median words')
        groups = {}
        for condition in ['neutral', 'angry', *[t['condition'] for t in plan['targets']]]:
            group = [r for r in chosen if r['condition'] == condition]
            groups[condition] = {(r['question_id'], r['voice']): r for r in group}
            if not group:
                continue
            n = len(group)
            counts = {m: sum(r['harness'][m]['correct'] for r in group) for m in values}
            counts['strict-format-extractable'] = sum(r['strict-format-extractable'] for r in group)
            lengths = [len(r['response'].split()) for r in group]
            print(f'{condition:31} {n:3d} {counts["flexible-extract"]/n:8.1%} {counts["strict-match"]/n:7.1%}'
                  f' {counts["strict-format-extractable"]/n:12.1%} {statistics.mean(lengths):11.1f}'
                  f' {statistics.median(lengths):13.1f}')
            summaries.append(dict(subset=subset, condition=condition, n=n, correct=counts,
                mean_words=statistics.mean(lengths), median_words=statistics.median(lengths)))
        for metric in ('flexible-extract', 'strict-match', 'strict-format-extractable'):
            for condition, group in groups.items():
                if condition == 'neutral':
                    continue
                keys = sorted(group.keys() & groups['neutral'].keys())
                if not keys:
                    continue
                def correct(row):
                    return row[metric] if metric == 'strict-format-extractable' else row['harness'][metric]['correct']
                result = paired([(correct(groups['neutral'][k]), correct(group[k])) for k in keys])
                comparisons.append(dict(subset=subset, metric=metric, condition=condition, **result))
                print(f'{metric} / {condition}: matched N={len(keys)}; gain={result["gain_pp"]:+.2f} pp; '
                      f'W/L={result["wins"]}/{result["losses"]}; p={result["mcnemar_exact_p"]:.4f}; CI={result["ci95_gain_pp"]}')
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    (output/'summary.json').write_text(json.dumps(dict(primary_metric='flexible-extract', prompt=plan['prompt'],
        selection=plan['selection'], accuracy=summaries, paired=comparisons, scorer=scorer.provenance,
        caveat=plan['caveat']+' P-values and paired bootstrap intervals are exploratory/unadjusted.'), indent=2)+'\n')
    print('\n'+plan['caveat'])
    print('Partial rows can have different N; paired comparisons always use identical neutral questions.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['prepare', 'evaluate', 'summary'], default='prepare')
    p.add_argument('--recovery', type=Path, default=regression.DEFAULT_RECOVERY)
    p.add_argument('--regressions', type=Path, default=Path('outputs/kimi-bbh3-regressions-L16-L17-L18'))
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--model-source', type=Path, default=Path('external/Kimi-Audio'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        report(args.output); return
    with (args.output/'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == 'prepare':
            plan = prepare(args)
            reuse_answers(plan, args.output, [args.recovery, args.regressions])
            done = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(args.output/'results.jsonl')}
            pending = sum((q['question_id'], q['voice'], t['condition']) not in done for q in plan['pairs'] for t in plan['targets'])
            print(f'{len(plan["pairs"])} original neutral questions × 3 layers; {pending} additional answers pending. OLD prompt.', flush=True)
        else:
            plan = json.loads((args.output/'run_config.json').read_text())
            regression.evaluate(plan, args)


if __name__ == '__main__':
    main()
