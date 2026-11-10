"""Run the agreed Jenny steering experiments sequentially; report them without loading models."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from compare_cloud_accuracy import PROMPT
from steer import summarize

ROOT = Path('outputs/jenny-large')
SOURCE = Path('outputs/steer-jenny-angry-complete')
COLLECTION = Path('outputs/jenny-svd-probe')
TARGETS = [(11, 'answer', 1), (12, 'acoustic', 1), (16, 'answer', 1.25)]
SCHEDULES = ['early', 'middle', 'late', 'first_forward', 'periodic2']


def conditions(schedules=('always',)):
    result = [dict(name='baseline', interventions=[])]
    for schedule in schedules:
        for layer, group, strength in TARGETS:
            name = f'L{layer}-{group}-a{strength:g}'
            if schedule != 'always':
                name += '-'+schedule
            row = dict(name=name, interventions=[dict(layer=layer, group=group, strength=strength)])
            if schedule != 'always':
                row['schedule'] = schedule
            result.append(row)
    return result


def catalog():
    return [
        ('Original direction, literal seed42 confirmation', Path('outputs/steer-jenny-original-vector-L12-seed42/original_mean'),
         [c for c in conditions() if c['name'] in ('baseline', 'L12-acoustic-a1')]),
        ('Extract only angry-correct pairs', Path('outputs/steer-jenny-outcome-filters/matched/angry_correct_mean'), conditions()),
        ('Extract all except neutral-correct / angry-wrong pairs', Path('outputs/steer-jenny-outcome-filters/matched/exclude_losses_mean'), conditions()),
        ('Directions by unmasking phase within each 32-token decoding block', ROOT/'temporal-evaluation/phase', conditions()),
        ('Directions by global 20-forward windows', ROOT/'temporal-evaluation/forward_windows', conditions()),
        ('Original direction with timing schedules', ROOT/'timing/original_mean', conditions(['always', *SCHEDULES])),
    ]


def live_rows(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines)-1 and not line.endswith('\n'):
                break
            raise
    return rows


def report():
    current = ROOT/'current-stage.json'
    if current.exists():
        stage = json.loads(current.read_text())
        print(f'Queue: {stage["status"]} — {stage["stage"]} ({stage["time"]})')
    pairs = ROOT/'temporal-vectors/pairs'
    print(f'Temporal collection: {len(list(pairs.glob("*.pt"))) if pairs.exists() else 0}/425 pairs saved.')
    for title, folder, planned in catalog():
        rows = live_rows(folder/'results.jsonl')
        latest = {(r['clip_id'], r['condition']): r for r in rows}
        expected = 460*len(planned)
        completed = sum(name in {c['name'] for c in planned} for _, name in latest)
        status = 'complete' if completed == expected else 'partial' if completed else 'not started'
        print(f'\n{title}: {status} ({completed}/{expected} answers)\n  {folder}', flush=True)
        summary = summarize(rows, planned)
        present = {r['condition'] for r in summary}
        for c in planned[1:]:
            if c['name'] not in present:
                print(f'  {c["name"]}: no paired answers yet')


def mark(stage, status):
    value = dict(stage=stage, status=status, time=datetime.now(timezone.utc).isoformat())
    temporary = ROOT/'current-stage.tmp'
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(ROOT/'current-stage.json')


def run(stage, script, *args):
    command = [sys.executable, '-u', script, *map(str, args)]
    mark(stage, 'running')
    print(f'\n{datetime.now(timezone.utc).isoformat()} {stage}\n{" ".join(command)}', flush=True)
    child = subprocess.Popen(command, start_new_session=True)
    try:
        code = child.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def reuse_always(folder, vector, sources):
    """Copy only proven identical checkpoint, cohort and literal-seed42 answers."""
    destination = json.loads((folder/'run_config.json').read_text())
    expected = {r['clip_id']: r for r in destination['recordings']}
    stat = vector.stat()
    saved = {(r['clip_id'], r['condition']): r for r in live_rows(folder/'results.jsonl')}
    added = 0
    with (folder/'results.jsonl').open('a') as stream:
        for source in sources:
            config_path = source/'run_config.json'
            if not config_path.exists():
                continue
            config = json.loads(config_path.read_text())
            settings = config['settings']
            if (Path(settings['vectors']).resolve() != vector.resolve()
                    or settings.get('vector_bytes') != stat.st_size
                    or settings.get('vector_mtime_ns') != stat.st_mtime_ns
                    or settings.get('seed_policy') != 'constant_before_every_generation'
                    or settings.get('mode', 'fixed') != 'fixed'):
                continue
            source_records = {r['clip_id']: r for r in config['recordings']}
            allowed = {c['name']: c for c in conditions()}
            for row in live_rows(source/'results.jsonl'):
                cid, name = row['clip_id'], row['condition']
                if (cid not in expected or name not in allowed or (cid, name) in saved
                        or source_records.get(cid) != expected[cid]):
                    continue
                if (row.get('seed') != 42 or row.get('prompt') != PROMPT or row.get('max_new_tokens') != 512
                        or row.get('decoder') != 'fast_dllm' or row.get('mode') != 'fixed'
                        or row.get('schedule', 'always') != 'always'
                        or row.get('interventions') != allowed[name]['interventions']
                        or str(row['reference_answer']) != str(expected[cid]['answer'])):
                    continue
                stream.write(json.dumps(row)+'\n')
                saved[cid, name] = row
                added += 1
    print(f'Reused {added} compatible original-checkpoint seed42 answers.', flush=True)


def evaluate(stage, vector, method, output, condition_file=None, prepare=False):
    args = ['--vectors', vector, '--methods', method, '--output', output, '--include-l11']
    if condition_file:
        args += ['--conditions-file', condition_file]
    if prepare:
        args += ['--prepare-only']
    run(stage, 'evaluate_audio_steering.py', *args)


def clock_smoke(vector):
    folder = ROOT/'clock-smoke'
    source = json.loads((SOURCE/'run_config.json').read_text())
    originals = {r['clip_id']: r for r in live_rows(catalog()[0][1]/'results.jsonl') if r['condition'] == 'baseline'}
    records = [r for r in source['recordings'] if r['clip_id'] in originals][:2]
    if len(records) != 2:
        raise ValueError('Need two seed42 baselines before the progress-observer smoke test')
    zero = dict(name='clock-zero', schedule='periodic2', interventions=[dict(layer=12, group='acoustic', strength=0)])
    plan = ROOT/'clock-smoke-plan.json'
    plan.write_text(json.dumps(dict(recordings=records, conditions=[dict(name='baseline', interventions=[]), zero]), indent=2)+'\n')
    run('Verify observational diffusion clock (zero steering)', 'steer_audio.py', '--plan', plan,
        '--vectors', vector, '--output', folder)
    rows = {(r['clip_id'], r['condition']): r for r in live_rows(folder/'results.jsonl')}
    for recording in records:
        cid = recording['clip_id']
        for condition in ('baseline', 'clock-zero'):
            row, original = rows[cid, condition], originals[cid]
            if (row['response'], row['forwards']) != (original['response'], original['forwards']):
                raise ValueError(f'Clock smoke changed generation for {cid}; stopping before temporal collection')
    print('Clock smoke passed: exact responses and forward counts unchanged.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary-only', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    if args.summary_only:
        report()
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    original_config = json.loads((SOURCE/'run_config.json').read_text())
    vector = Path(original_config['settings']['vectors'])
    if not vector.is_file():
        p.error(f'Original checkpoint missing at {vector}; restore it before starting')
    if len(original_config['recordings']) != 460:
        p.error('Expected the unchanged 460-recording evaluation cohort')
    collection = json.loads((COLLECTION/'collection-config.json').read_text())
    if len(collection['pairs']) != 425 or collection.get('averaging', 'matched') != 'matched':
        p.error('Expected the first-set 425-pair matched-forward collection')
    if ({r['question_id'] for r in original_config['recordings']} &
            {r['question_id'] for r in collection['manifest'].values()}):
        p.error('Extraction and evaluation questions overlap')
    timing = ROOT/'timing-conditions.json'
    timing.write_text(json.dumps(conditions(['always', *SCHEDULES]), indent=2)+'\n')
    if args.prepare_only:
        print('Original vector:', vector)
        print('Extraction: first-set 425 matched pairs. Evaluation: unchanged second-set 460 neutrals.')
        print('Three independent targets; matched averaging only; seed42 before each generation.')
        report()
        return
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        run('Confirm original L12 with literal seed42', 'evaluate_audio_steering.py', '--vectors', vector,
            '--methods', 'original_mean', '--conditions', 'L12-acoustic-a1',
            '--output', 'outputs/steer-jenny-original-vector-L12-seed42')
        clock_smoke(vector)
        for selection, method in [('angry_correct', 'angry_correct_mean'), ('exclude_losses', 'exclude_losses_mean')]:
            run('Extract '+method, 'extract_benefit_vectors.py', '--collection', COLLECTION,
                '--filter', selection, '--output', 'outputs/jenny-outcome-filters/matched')
            evaluate('Evaluate '+method, Path('outputs/jenny-outcome-filters/matched')/method/'style-vectors.pt',
                     method, Path('outputs/steer-jenny-outcome-filters/matched'))
        run('Collect both temporal methods once', 'collect_temporal_vectors.py', '--collection', COLLECTION,
            '--original-vector', vector, '--output', ROOT/'temporal-vectors')
        for method in ('phase', 'forward_windows'):
            evaluate('Evaluate '+method, ROOT/'temporal-vectors'/method/'style-vectors.pt', method, ROOT/'temporal-evaluation')
        evaluate('Prepare original-vector schedules', vector, 'original_mean', ROOT/'timing', timing, prepare=True)
        reuse_always(ROOT/'timing/original_mean', vector, [catalog()[0][1],
            Path('outputs/steer-jenny-L16-full460-seed42'), Path('outputs/steer-jenny-L12-vs-L16-full460-seed42'), SOURCE])
        evaluate('Evaluate original-vector schedules', vector, 'original_mean', ROOT/'timing', timing)
    except BaseException:
        current = json.loads((ROOT/'current-stage.json').read_text())
        mark(current['stage'], 'stopped; rerun the same command to resume')
        raise
    mark('All requested experiments', 'finished')
    report()


if __name__ == '__main__':
    main()
