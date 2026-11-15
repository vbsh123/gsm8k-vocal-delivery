"""Rescore BBH-three and SVAMP saved answers through installed task APIs; preserve original files."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from rescore_audio_steering import HarnessTaskScorer, numeric_equal, numeric_within_tolerance, paired
from score_bbh_harness import report_saved
from steer import read_rows
from steer_bbh_mean import summarize as steering_summary
from run_svamp500 import SOURCE_SHA256 as SVAMP_SHA, select_questions


def compare(previous, current, destination):
    names = {'harness_strict': 'strict-match', 'harness_flexible': 'flexible-extract',
             'flexible_numeric': 'flexible-numeric', 'harness_format_compliant': 'format-compliant'}
    old = {}
    for row in previous.get('accuracy', []):
        if isinstance(row.get('correct'), dict):
            for metric, correct in row['correct'].items():
                if metric in names:
                    old[names[metric], row['condition']] = (row['n'], correct)
        else:
            old[row['metric'], row['condition']] = (row['n'], row['correct'])
    checks = []
    for row in current['accuracy']:
        key = row['metric'], row['condition']
        if key in old:
            n, correct = old[key]
            checks.append(dict(metric=key[0], condition=key[1], previous_n=n, current_n=row['n'],
                previous_correct=correct, current_correct=row['correct'],
                same_cohort_size=n == row['n'], same_correct_count=correct == row['correct']))
    (destination/'comparison.json').write_text(json.dumps(checks, indent=2)+'\n')
    equal = all(c['same_cohort_size'] and c['same_correct_count'] for c in checks)
    print(f'Comparison with saved summary: {len(checks)} matched rows; '
          + ('all counts agree.' if checks and equal else 'inspect comparison.json (missing, changed cohort or score difference).'), flush=True)


def previous_summary(path, destination):
    value = json.loads(path.read_text()) if path.exists() else {}
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = destination/'previous-summary.json'
    if value and not snapshot.exists():
        snapshot.write_text(json.dumps(value, indent=2)+'\n')
    return value


def audit_svamp(run, source, destination):
    old = previous_summary(run/'summary.json', destination)
    bank = {r['question_id']: r for r in read_rows(source/'questions.jsonl')}
    original = (source/'SVAMP-source.json').read_bytes()
    if hashlib.sha256(original).hexdigest() != SVAMP_SHA:
        raise ValueError('Original SVAMP source checksum differs')
    if bank != {q['question_id']: q for q in select_questions(original, 500)}:
        raise ValueError('Saved SVAMP questions differ from the fixed original 500')
    latest = {}
    for row in read_rows(run/'results.jsonl'):
        if row['condition'] not in ('neutral', 'angry'):
            continue
        key = row['question_id'], row['voice'], row['condition']
        if key in latest and latest[key]['response'] != row['response']:
            raise ValueError('Conflicting SVAMP saved responses')
        if row['reference_answer'] != bank[key[0]]['answer']:
            raise ValueError('Saved SVAMP reference differs from source')
        latest[key] = row
    matched = {k[:2] for k in latest if k[2] == 'neutral'} & {k[:2] for k in latest if k[2] == 'angry'}
    rows = [latest[q, v, c] for q, v in sorted(matched) for c in ('neutral', 'angry')]
    scorer = HarnessTaskScorer('gsm8k', destination/'upstream')
    values = scorer.score([r['response'] for r in rows],
        [dict(question=bank[r['question_id']]['question'], answer=r['reference_answer']) for r in rows])
    rescored = []; disagreements = []
    for i, row in enumerate(rows):
        scores = {name: result[i]['correct'] for name, result in values.items()}
        predictions = {name: result[i]['prediction'] for name, result in values.items()}
        scores['flexible-numeric'] = numeric_equal(predictions['flexible-extract'], row['reference_answer'])
        scores['author-tolerance'] = numeric_within_tolerance(predictions['flexible-extract'], row['reference_answer'])
        scores['format-compliant'] = predictions['strict-match'] != '[invalid]'
        entry = dict(row, api_scores=scores, api_predictions=predictions)
        rescored.append(entry)
        previous = {name: row[field] for field, name in [('harness_strict', 'strict-match'),
            ('harness_flexible', 'flexible-extract'), ('flexible_numeric', 'flexible-numeric')]
            if field in row}
        if any(scores[name] != value for name, value in previous.items()):
            disagreements.append(dict(entry, previous_scores=previous))
    summary = dict(scorer=scorer.provenance, accuracy=[], paired=[])
    print(f'\nSVAMP: {len(matched)} matched pairs; installed GSM8K task scoring on SVAMP references.', flush=True)
    print('No dedicated SVAMP task exists in this pinned harness. Numeric equivalence is additional.', flush=True)
    print('author-tolerance: extracted numeric answer within 0.1 of gold; authors\' numeric criterion, not their equation parser.', flush=True)
    for metric in ('flexible-numeric', 'flexible-extract', 'strict-match', 'format-compliant', 'author-tolerance'):
        groups = {c: {(r['question_id'], r['voice']): r['api_scores'][metric]
                      for r in rescored if r['condition'] == c} for c in ('neutral', 'angry')}
        for c, answers in groups.items():
            n = len(answers); correct = sum(answers.values())
            summary['accuracy'].append(dict(metric=metric, condition=c, n=n, correct=correct))
            print(f'{metric} / {c}: {correct}/{n}' + (f' ({correct/n:.1%})' if n else ''))
        if matched:
            result = paired([(groups['neutral'][k], groups['angry'][k]) for k in sorted(matched)])
            summary['paired'].append(dict(metric=metric, condition='angry', **result))
            lo, hi = result['ci95_gain_pp']
            print(f'  gain={result["gain_pp"]:+.2f} pp; W/L={result["wins"]}/{result["losses"]}; '
                  f'p={result["mcnemar_exact_p"]:.4f}; CI=[{lo:+.1f}, {hi:+.1f}]')
    (destination/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for name, entries in [('rescored.jsonl', rescored), ('disagreements.jsonl', disagreements)]:
        (destination/name).write_text(''.join(json.dumps(r)+'\n' for r in entries))
    print(f'Per-answer disagreements against stored numeric/harness scores: {len(disagreements)}', flush=True)
    compare(old, summary, destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('outputs/harness-api-audit'))
    p.add_argument('--bbh-source', type=Path, default=Path('outputs/azure-bbh-logical200'))
    p.add_argument('--bbh-deliveries', type=Path, default=Path('outputs/bbh-logical200-diffa2-harness-chunked'))
    p.add_argument('--bbh-steering-source', type=Path, default=Path('outputs/azure-bbh-logical250-steering'))
    p.add_argument('--bbh-steering', type=Path, default=Path('outputs/bbh-logical250-angry-mean'))
    p.add_argument('--svamp-source', type=Path, default=Path('outputs/azure-svamp500'))
    p.add_argument('--svamp-run', type=Path, default=Path('outputs/svamp500-diffa2-hashes'))
    args = p.parse_args()
    completed = []
    if (args.bbh_deliveries/'results.jsonl').exists():
        destination = args.output/'bbh-three-deliveries'
        old = previous_summary(args.bbh_deliveries/'harness/summary.json', destination)
        current = report_saved(args.bbh_deliveries, args.bbh_source, destination=destination)
        compare(old, current, destination); completed.append('bbh-three-deliveries')
    if (args.bbh_steering/'results.jsonl').exists():
        destination = args.output/'bbh-three-steering'
        old = previous_summary(args.bbh_steering/'summary.json', destination)
        current = steering_summary(SimpleNamespace(output=args.bbh_steering, source=args.bbh_steering_source), destination=destination)
        compare(old, current, destination); completed.append('bbh-three-steering')
    if (args.svamp_run/'results.jsonl').exists():
        audit_svamp(args.svamp_run, args.svamp_source, args.output/'svamp-hashes'); completed.append('svamp-hashes')
    if not completed:
        p.error('No saved BBH-three or SVAMP runs found at the supplied paths')
    print('\nFinished CPU rescoring:', ', '.join(completed), flush=True)
    print('Original responses and summaries retained. No models loaded or generations run.', flush=True)


if __name__ == '__main__':
    main()
