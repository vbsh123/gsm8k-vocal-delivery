"""Run >=40%-recovery ten-question screen candidates on all 74 neutrals."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from steer import read_rows, summarize
from archive.steering.steer_jenny_recovery import correct, recovery_summary
from archive.steering.steer_jenny_six import main as run


def select_conditions(source, screen, rows, minimum=.4):
    screened = screen['recordings']
    conditions = screen['settings']['conditions']
    latest = {(r['clip_id'], r['condition']): r for r in rows}
    selected, evidence = [dict(name='baseline', interventions=[])], []
    for c in conditions:
        if c['name'] == 'baseline':
            continue
        recovered = sum(correct(latest[r['clip_id'], c['name']]) for r in screened)
        if recovered / len(screened) >= minimum:
            selected.append(c)
            evidence.append(dict(condition=c['name'], recovered=recovered, n=len(screened)))
    return selected, evidence


def report_unseen(output, screen_ids):
    config = json.loads((output/'run_config.json').read_text())
    latest = {(r['clip_id'], r['condition']): r for r in read_rows(output/'results.jsonl')}
    remaining = [r for r in config['recordings'] if r['question_id'] not in screen_ids]
    print(f'\nRemaining {len(remaining)} questions not used to choose these conditions')
    print(f'{"Condition":25} {"Paired":>7} {"Baseline failed":>16} {"Recovered":>10} {"Recovery":>10}')
    for c in config['settings']['conditions'][1:]:
        pairs = [(latest[r['clip_id'], 'baseline'], latest[r['clip_id'], c['name']]) for r in remaining
                 if (r['clip_id'], 'baseline') in latest and (r['clip_id'], c['name']) in latest]
        failures = [(b, s) for b, s in pairs if not correct(b)]
        recovered = sum(correct(s) for b, s in failures)
        rate = f'{recovered/len(failures):.1%}' if failures else 'n/a'
        print(f'{c["name"]:25} {len(pairs):7} {len(failures):16} {recovered:10} {rate:>10}')
    print('The ten screening questions are excluded here; the remaining cohort is still outcome-selected.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=Path('outputs/steer-jenny-angry-recovery74-expanded'))
    parser.add_argument('--screen-run', type=Path, default=Path('outputs/steer-jenny-recovery10-newlayers-a1.25-seed42'))
    parser.add_argument('--vectors', type=Path, default=Path('outputs/jenny-angry-vectors/style-vectors.pt'))
    parser.add_argument('--output', type=Path, default=Path('outputs/steer-jenny-recovery74-promoted-a1.25-seed42'))
    parser.add_argument('--min-recovery', type=float, default=.4)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()
    if args.summary_only:
        saved = json.loads((args.output/'selected-plan.json').read_text())
        recovery_summary(args.output)
        report_unseen(args.output, set(saved['screen_question_ids']))
        return
    source_path, screen_path = args.source_run/'run_config.json', args.screen_run/'run_config.json'
    source, screen = json.loads(source_path.read_text()), json.loads(screen_path.read_text())
    screen_results = args.screen_run/'results.jsonl'
    try:
        conditions, evidence = select_conditions(source, screen, read_rows(screen_results), args.min_recovery)
    except ValueError as error:
        parser.error(str(error))
    stat = args.vectors.stat()
    for previous in [source, screen]:
        settings = previous['settings']
        if (args.vectors.resolve() != Path(settings['vectors']).resolve()
                or stat.st_size != settings['vector_bytes'] or stat.st_mtime_ns != settings['vector_mtime_ns']):
            parser.error('Vector file differs from the source/screen experiment')
        if settings['seed'] != 42 or settings['mode'] != 'fixed' or settings['style'] != 'angry':
            parser.error('Requires the same fixed angry-vector experiment with seed42')
    recordings = source['recordings']
    extraction = {r['question_id'] for r in read_rows(args.vectors.parent/'extraction-questions.jsonl')}
    if extraction & {r['question_id'] for r in recordings}:
        parser.error('Recovery recordings overlap vector extraction')
    screen_ids = sorted(r['question_id'] for r in screen['recordings'])
    plan = dict(recordings=recordings, conditions=conditions, screening=evidence,
                screen_question_ids=screen_ids, minimum_recovery=args.min_recovery,
                selection='Full original74 neutral recovery recordings; targets selected on ten of them',
                source_run=str(args.source_run.resolve()), screen_run=str(args.screen_run.resolve()),
                source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                screen_sha256=hashlib.sha256(screen_path.read_bytes()).hexdigest(),
                screen_results_sha256=hashlib.sha256(screen_results.read_bytes()).hexdigest(), seed=42)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/'selected-plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        parser.error('Frozen promotion plan changed; choose a new output directory')
    path.write_text(json.dumps(plan, indent=2)+'\n')
    print(f'{len(conditions)-1} conditions meet >= {args.min_recovery:.0%} recovery on the ten-question screen.')
    print(f'74 neutral recordings × {len(conditions)} conditions including fresh baseline = {74*len(conditions)} generations.')
    print('Every intervention runs individually at strength1.25 with angry-minus-neutral vectors, seed42.')
    print('Reports will show all74 and the remaining64 separately.')
    command = ['--previous', str(args.source_run), '--vectors', str(args.vectors),
               '--prepared-plan', str(path), '--output', str(args.output)]
    if args.prepare_only:
        command.append('--prepare-only')
    with patch('archive.steering.steer_jenny_six.report', side_effect=lambda rows, cs, rs: summarize(rows, cs)):
        run(command)
    if not args.prepare_only:
        recovery_summary(args.output)
        report_unseen(args.output, set(screen_ids))


if __name__ == '__main__':
    main()
