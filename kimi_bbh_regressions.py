"""Freeze recovery-selected Kimi layer choices; test the original regression cases.

Reuses the existing 125-pair audio/answer vectors. No collection or base model.
Outcome-selected diagnostics, not an estimate of population accuracy.
"""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

import kimi_bbh_layer_diagnostic as diagnostic
from steer import read_rows

DEFAULT_RECOVERY = Path('outputs/kimi-bbh3-collect125-audio-answer-steer30')
DEFAULT_OUTPUT = Path('outputs/kimi-bbh3-regressions')


def regressions(rows):
    return [p for p in diagnostic.matched_pairs(rows)
            if p['neutral']['harness']['flexible-extract']['correct']
            and not p['angry']['harness']['flexible-extract']['correct']]


def parse_targets(values):
    targets = []
    for value in values:
        group, layer = value.split(':')
        layer = int(layer)
        if group not in diagnostic.GROUPS or layer not in diagnostic.LAYERS:
            raise ValueError(f'Expected audio:0..27 or answer:0..27, got {value}')
        target = dict(layer=layer, group=group, strength=1., condition=diagnostic.condition_name(layer, group))
        if target not in targets:
            targets.append(target)
    return targets


def complete_recovery_rows(plan, rows):
    expected = {(p['question_id'], p['voice'], diagnostic.condition_name(layer, group))
                for p in plan['pairs'] for layer in plan['layers'] for group in plan['groups']}
    actual = {}
    for row in rows:
        key = row['question_id'], row['voice'], row['condition']
        if key in expected:
            if key in actual:
                raise ValueError(f'Duplicate recovery sweep answer: {key}')
            if row['prompt'] != plan['prompt'] or row['seed'] != plan['seed']:
                raise ValueError('Recovery sweep settings differ from its frozen configuration.')
            actual[key] = row
    return expected, actual


def rank_targets(plan, rows, scores, top_per_group):
    """Rank only recovery outputs; regression outcomes never enter selection."""
    ranking = []
    for group in plan['groups']:
        candidates = []
        for layer in plan['layers']:
            condition = diagnostic.condition_name(layer, group)
            indices = [i for i, row in enumerate(rows) if row['condition'] == condition]
            if len(indices) != len(plan['pairs']):
                raise ValueError('Layer choice requires the completed recovery sweep.')
            candidates.append(dict(layer=layer, group=group, condition=condition, strength=1.,
                n=len(indices), flexible_correct=sum(scores['flexible-extract'][i]['correct'] for i in indices),
                strict_correct=sum(scores['strict-match'][i]['correct'] for i in indices)))
        candidates.sort(key=lambda r: (-r['flexible_correct'], -r['strict_correct'], r['layer']))
        ranking.extend(candidates[:top_per_group])
    return ranking


def prepare(args):
    from score_bbh_harness import HarnessScorer
    recovery = json.loads((args.recovery/'run_config.json').read_text())
    if recovery.get('protocol') != diagnostic.PROTOCOL:
        raise ValueError('Expected the corrected 125-collection audio+answer recovery experiment.')
    config = args.output/'run_config.json'
    if config.exists():
        plan = json.loads(config.read_text())
        if plan['recovery_directory'] != str(args.recovery.absolute()):
            raise ValueError('Recovery source changed. Use a new output directory.')
        if args.targets and parse_targets(args.targets) != plan['targets']:
            raise ValueError('Layer choices already frozen. Use a new output directory to change them.')
        if diagnostic.digest(Path(plan['vector_source'])) != plan['vector_sha256']:
            raise ValueError('Collected vectors changed after selection.')
        print('Reusing frozen regression targets:', plan['targets'], flush=True)
        return plan
    while True:
        expected, actual = complete_recovery_rows(recovery, read_rows(args.recovery/'results.jsonl'))
        if expected <= actual.keys():
            break
        if not args.wait:
            raise ValueError(f'Recovery sweep incomplete: {len(actual)}/{len(expected)}. Use --wait or finish it first.')
        print(f'Waiting for recovery sweep: {len(actual)}/{len(expected)} steered answers saved.', flush=True)
        time.sleep(30)
    benchmark = Path(recovery['benchmark'])
    original = json.loads((benchmark/'run_config.json').read_text())
    if diagnostic.digest(benchmark/'run_config.json') != recovery['source_config_sha256']:
        raise ValueError('Original benchmark configuration changed after vector collection.')
    if original['prompt'] != recovery['prompt']:
        raise ValueError('Original benchmark prompt differs from the collected vectors.')
    vectors = args.recovery/'style-vectors.pt'
    targets = parse_targets(args.targets) if args.targets else None
    selection = 'Explicit layer/group choices supplied before regression evaluation.'
    if targets is None:
        rows = [actual[key] for key in sorted(expected)]
        scorer = HarnessScorer(args.output/'selection-harness/upstream')
        docs = [dict(input=original['bank'][r['question_id']]['original_input'],
                     target=original['bank'][r['question_id']]['original_target']) for r in rows]
        scores = scorer.score([r['response'] for r in rows], docs)
        ranking = rank_targets(recovery, rows, scores, args.top_per_group)
        targets = [{k: r[k] for k in ('layer', 'group', 'strength', 'condition')} for r in ranking]
        (args.output/'recovery-ranking.json').write_text(json.dumps(ranking, indent=2)+'\n')
        selection = (f'Top {args.top_per_group} per group from COMPLETE recovery sweep; rank by flexible correct, '
                     'then strict correct, then ascending layer index. Frozen before regression generation.')
    pairs = regressions(read_rows(benchmark/'rescored.jsonl'))
    if not pairs:
        raise ValueError('No flexible-score neutral-correct / angry-wrong regression pairs.')
    collection_ids = {p['question_id'] for p in recovery['collection_pairs']}
    for pair in pairs:
        pair['in_vector_collection'] = pair['question_id'] in collection_ids
        for style in ('neutral', 'angry'):
            row = pair[style]
            if row['prompt'] != recovery['prompt'] or row['seed'] != 42 or row['max_new_tokens'] != 512:
                raise ValueError('Regression reference generation settings differ from the vector recipe.')
            if diagnostic.digest(row['audio_file']) != row['audio_sha256']:
                raise ValueError(f'Audio missing/changed: {row["audio_file"]}')
    overlap = [p['question_id'] for p in pairs if p['in_vector_collection']]
    plan = dict(protocol='kimi-bbh3-frozen-layers-regressions-v1', recovery_directory=str(args.recovery.absolute()),
        recovery_config_sha256=diagnostic.digest(args.recovery/'run_config.json'),
        recovery_results_sha256=diagnostic.digest(args.recovery/'results.jsonl'),
        benchmark=str(benchmark), bank={p['question_id']: original['bank'][p['question_id']] for p in pairs},
        prompt=recovery['prompt'], seed=42, generation=recovery['generation'],
        instruct_model=recovery['instruct_model'], instruct_revision=recovery['instruct_revision'],
        source_revision=recovery['source_revision'], vector_source=str(vectors.absolute()),
        vector_sha256=diagnostic.digest(vectors), collection_ids=sorted(collection_ids),
        overlap_with_collection=overlap, pairs=pairs, targets=targets, selection=selection,
        caveat='Outcome-selected regression diagnostic. Collection overlap is explicitly marked and reported separately. No recollection, base steering, or overall accuracy claim.')
    diagnostic.freeze(config, plan)
    print(f'{len(pairs)} regression questions; {len(overlap)} seen in vector collection; '
          f'{len(pairs)-len(overlap)} collection-unseen. Targets: {targets}', flush=True)
    return plan


def evaluate(plan, args):
    import torch
    if diagnostic.digest(Path(plan['vector_source'])) != plan['vector_sha256']:
        raise ValueError('Collected vectors changed.')
    path = args.output/'results.jsonl'
    done = {(r['question_id'], r['voice'], r['condition']) for r in read_rows(path, repair=True)}
    pending = [(p, t) for t in plan['targets'] for p in plan['pairs']
               if (p['question_id'], p['voice'], t['condition']) not in done]
    with path.open('a') as stream:
        for pair in plan['pairs']:
            for style in ('neutral', 'angry'):
                key = pair['question_id'], pair['voice'], style
                if key not in done:
                    stream.write(json.dumps(dict(pair[style], condition=style,
                        in_vector_collection=pair['in_vector_collection'], reused_from=plan['benchmark']))+'\n')
                    done.add(key)
    print(f'{len(pending)} {plan.get("evaluation_label", "regression")} generations pending; no new collection or base checkpoint.', flush=True)
    if not pending:
        return
    vectors = torch.load(plan['vector_source'], map_location='cpu', weights_only=True)
    if vectors['protocol'] != diagnostic.PROTOCOL:
        raise ValueError('Expected the audio+answer vectors from the corrected 125-pair collection.')
    if set(vectors['question_ids']) != set(plan['collection_ids']):
        raise ValueError('Vector collection IDs differ from the frozen source.')
    with Path('outputs/jenny-task-vectors.lock').open('w') as lock:
        print('Waiting for shared GPU lock.', flush=True); fcntl.flock(lock, fcntl.LOCK_EX)
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            print('Waiting for active GPU work.', flush=True); time.sleep(30)
        model = diagnostic.load_model(args.model_source)
        with path.open('a') as stream:
            for i, (pair, target) in enumerate(pending, 1):
                row = pair['neutral']
                if diagnostic.digest(row['audio_file']) != row['audio_sha256']:
                    raise ValueError('Evaluation audio changed.')
                vector = vectors['audio_emotion' if target['group'] == 'audio' else 'emotion'][target['layer']]
                print(f'[{i}/{len(pending)}] {pair["question_id"]} {target["condition"]}', flush=True)
                started = time.monotonic()
                response = diagnostic.generate_steered(model, row, plan['prompt'], target['layer'], target['group'], vector)
                saved = dict(row, condition=target['condition'], response=response,
                    generation_seconds=time.monotonic()-started, interventions=[target],
                    in_vector_collection=pair['in_vector_collection'], reused_from=None)
                saved.pop('harness', None); saved.pop('strict-format-extractable', None)
                stream.write(json.dumps(saved)+'\n'); stream.flush()


def report(output):
    from score_bbh_harness import HarnessScorer
    from evaluate_bbh_audio_models import strict_format_bits
    plan = json.loads((output/'run_config.json').read_text())
    rows = read_rows(output/'results.jsonl')
    if not rows:
        print('No regression answers saved yet.'); return
    scorer = HarnessScorer(output/'harness/upstream')
    docs = [dict(input=plan['bank'][r['question_id']]['original_input'],
                 target=plan['bank'][r['question_id']]['original_target']) for r in rows]
    scores = scorer.score([r['response'] for r in rows], docs)
    formats = strict_format_bits(scorer, [r['response'] for r in rows], docs)
    scored = [dict(r, harness={m: v[i] for m, v in scores.items()},
                   **{'strict-format-extractable': formats[i]}) for i, r in enumerate(rows)]
    summaries = []
    for subset in ('all-regressions', 'collection-unseen', 'collection-seen'):
        subset_rows = [r for r in scored if subset == 'all-regressions'
                       or bool(r['in_vector_collection']) == (subset == 'collection-seen')]
        print('\n'+subset+' — outcome-selected diagnostic; percentages only.')
        print('Condition                         N  Flexible retained  Strict  Valid format')
        for condition in ['neutral', 'angry', *[t['condition'] for t in plan['targets']]]:
            group = [r for r in subset_rows if r['condition'] == condition]
            if not group:
                continue
            n = len(group)
            counts = {m: sum(r['harness'][m]['correct'] for r in group) for m in scores}
            counts['strict-format-extractable'] = sum(r['strict-format-extractable'] for r in group)
            print(f'{condition:32} {n:3d} {counts["flexible-extract"]:3d}/{n:<3d}'
                  f' ({counts["flexible-extract"]/n:5.1%})  {counts["strict-match"]/n:6.1%}'
                  f'  {counts["strict-format-extractable"]/n:6.1%}')
            summaries.append(dict(subset=subset, condition=condition, n=n, correct=counts))
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    (output/'summary.json').write_text(json.dumps(dict(selection=plan['selection'],
        overlap_with_collection=plan['overlap_with_collection'], results=summaries,
        scorer=scorer.provenance, caveat=plan['caveat']), indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['prepare', 'evaluate', 'summary'], default='prepare')
    p.add_argument('--recovery', type=Path, default=DEFAULT_RECOVERY)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--model-source', type=Path, default=Path('external/Kimi-Audio'))
    p.add_argument('--targets', nargs='+', help='Explicit fixed choices, e.g. audio:12 answer:20. Otherwise select top per group on recoveries.')
    p.add_argument('--top-per-group', type=int, default=1)
    p.add_argument('--wait', action='store_true', help='Wait for the full 30 × 28 × 2 recovery sweep before freezing choices.')
    args = p.parse_args()
    if not 1 <= args.top_per_group <= 28:
        p.error('--top-per-group must be between 1 and 28')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'summary':
        report(args.output)
        return
    with (args.output/'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == 'prepare':
            prepare(args)
        else:
            plan = json.loads((args.output/'run_config.json').read_text())
            evaluate(plan, args)


if __name__ == '__main__':
    main()
