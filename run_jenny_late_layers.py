"""Sweep late-layer steering on both evaluation cohorts, excluding extraction questions."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from evaluate_audio_steering import report
from steer import read_rows, summarize


LABEL = ('Second Azure cohort, same neutral recordings as the original timing run; '
         'late-layer development sweep, without outcome or classifier filtering.')


def conditions():
    result = [dict(name='baseline', interventions=[])]
    for layer in (29, 30, 31):
        for group in ('acoustic', 'answer'):
            for strength in (0.5, 1):
                for schedule in ('always', 'periodic2'):
                    name = f'L{layer}-{group}-a{strength}'
                    if schedule != 'always':
                        name += '-' + schedule
                    result.append(dict(name=name, schedule=schedule, interventions=[
                        dict(layer=layer, group=group, strength=strength)]))
    return result


def forward20_conditions():
    return [dict(name='baseline', interventions=[]),
            dict(name='L12-acoustic-a1-periodic2', schedule='periodic2', interventions=[
                dict(layer=12, group='acoustic', strength=1)])]


def combined_report(root):
    combined = {}
    for method in ('original_mean', 'forward20'):
        results, seen_questions, planned, tested, vector_identity = [], set(), 0, None, None
        for cohort in (root, root/'third-cohort'):
            folder = cohort/method
            path = folder/'run_config.json'
            if not path.exists():
                continue
            config = json.loads(path.read_text())
            questions = {r['question_id'] for r in config['recordings']}
            if questions & seen_questions:
                raise ValueError('Overlapping evaluation cohorts cannot be pooled')
            seen_questions.update(questions)
            planned += len(config['recordings'])
            current = config['settings']['conditions']
            if tested is not None and current != tested:
                raise ValueError('Conditions differ between evaluation cohorts')
            identity = tuple(config['settings'].get(k) for k in ('vectors','vector_bytes','vector_mtime_ns','seed','mode'))
            if vector_identity is not None and identity != vector_identity:
                raise ValueError('Vector checkpoint or steering recipe differs between evaluation cohorts')
            vector_identity = identity
            tested = current
            results.extend(read_rows(folder/'results.jsonl'))
        if tested is not None:
            print(f'\nCOMBINED: {method}; {planned} planned recordings across available cohort configs.', flush=True)
            combined[method] = summarize(results, tested)
    return combined


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path,
                   default=Path('outputs/jenny-large/timing/original_mean'))
    p.add_argument('--vectors', type=Path, help='Override the original vector checkpoint path if moved.')
    p.add_argument('--forward20-source', type=Path,
                   default=Path('outputs/jenny-large/temporal-evaluation/forward_windows'))
    p.add_argument('--forward20-vectors', type=Path, help='Override the 20-forward checkpoint path if moved.')
    p.add_argument('--output', type=Path, default=Path('outputs/steer-jenny-late-layers'))
    p.add_argument('--third-source-run', type=Path,
                   default=Path('outputs/steer-jenny-third-promoted/cohort'))
    p.add_argument('--third-baseline-source', type=Path,
                   default=Path('outputs/steer-jenny-third-promoted/original'))
    p.add_argument('--third-only', action='store_true', help='Queue only the extra final-cohort recordings after an existing second-cohort run.')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--summary-only', action='store_true')
    args = p.parse_args()
    if args.summary_only:
        combined_report(args.output)
        print('\nOptional cohort breakdown follows.', flush=True)
        report(args.output, ['original_mean', 'forward20'], LABEL)
        report(args.output/'third-cohort', ['original_mean', 'forward20'],
               'Final Azure cohort, all eligible neutral recordings; no classifier or outcome filtering.')
        return
    source = json.loads((args.source_run / 'run_config.json').read_text())
    vector = args.vectors or Path(source['settings']['vectors'])
    if not vector.is_file():
        p.error(f'Missing original checkpoint: {vector}; pass --vectors if it moved.')
    args.output.mkdir(parents=True, exist_ok=True)
    file = args.output / 'conditions.json'
    file.write_text(json.dumps(conditions(), indent=2) + '\n')
    temporal = json.loads((args.forward20_source / 'run_config.json').read_text())
    temporal_vector = args.forward20_vectors or Path(temporal['settings']['vectors'])
    if not temporal_vector.is_file():
        p.error(f'Missing 20-forward checkpoint: {temporal_vector}; pass --forward20-vectors if it moved.')
    temporal_file = args.output / 'forward20-conditions.json'
    temporal_file.write_text(json.dumps(forward20_conditions(), indent=2) + '\n')
    third = json.loads((args.third_source_run/'run_config.json').read_text())
    if {r['question_id'] for r in source['recordings']} & {r['question_id'] for r in third['recordings']}:
        p.error('Evaluation cohorts overlap')
    cohorts = [(args.source_run, args.output, args.source_run, len(source['recordings']), LABEL),
               (args.third_source_run, args.output/'third-cohort', args.third_baseline_source,
                len(third['recordings']), 'Final Azure cohort, all eligible neutral recordings; no classifier or outcome filtering.')]
    if args.third_only:
        cohorts = cohorts[1:]
    commands = []
    for source_run, output, baseline, count, label in cohorts:
        for method, checkpoint, conditions_file in [('original_mean', vector, file),
                                                    ('forward20', temporal_vector, temporal_file)]:
            commands.append([sys.executable, '-u', 'evaluate_audio_steering.py',
                             '--source-run', str(source_run), '--vectors', str(checkpoint),
                             '--methods', method, '--conditions-file', str(conditions_file),
                             '--num-examples', str(count), '--output', str(output),
                             '--baseline-source', str(baseline), '--cohort-label', label])
    count = sum(c[3] for c in cohorts)
    print(f'{count} recordings × 25 interventions '
          '(24 late-layer conditions, then L12 acoustic forward20 + periodic2); '
          'compatible saved baselines are reused. Each condition runs separately, seed42.', flush=True)
    print('periodic2 uses forwards 1, 3, 5, ... within each decoding block.', flush=True)
    for queued in commands:
        subprocess.run(queued + ['--prepare-only'], check=True)
    if args.prepare_only:
        return
    print('Waiting for the existing Azure emotion/accuracy queue before taking the GPU lock.', flush=True)
    subprocess.run([sys.executable, '-u', 'wait_for_experiments.py',
                    '--kind', 'azure-emotions'], check=True)
    with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
        print('Waiting for the shared GPU lock.', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        while subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            print('Waiting for active GPU work.', flush=True)
            time.sleep(30)
        for queued in commands:
            subprocess.run(queued, check=True)
    summary = combined_report(args.output)
    (args.output/'combined-summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
