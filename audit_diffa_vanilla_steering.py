"""Read-only audit of vanilla DIFFA fitting aggregates and saved L31 steering."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

import torch
import diffa_gsm_vanilla as vanilla
from steer import read_rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root, emotion, regime):
    folder = root/emotion/regime
    config_path = folder/'run_config.json'
    plan = json.loads(config_path.read_text())
    state = torch.load(folder/'collection-stream.pt', map_location='cpu', weights_only=True)
    vector = torch.load(folder/'style-vectors.pt', map_location='cpu', weights_only=True)
    ids = state['completed_ids']
    require(state['run_config_sha256'] == vanilla.task.winners.digest(config_path), 'Checkpoint config hash differs.')
    require(ids == plan['collection_ids'] == vector['extraction_question_ids'], 'Fitting IDs differ or collection incomplete.')
    require([r['question_id'] for r in state['rows']] == ids, 'Checkpoint row order differs.')
    require(len(set(ids)) == len(ids), 'Duplicate fitting IDs.')
    require(vector['sign'] == f'{emotion} minus neutral', 'Vector sign metadata differs.')
    require(vector['prompt'] == plan['collection_prompt'] == plan['prompt'], 'Collection/evaluation prompt differs.')
    require(vector['max_new_tokens'] == plan['max_new_tokens'], 'Collection budget differs.')
    require(vector['groups'] == ['acoustic', 'semantic', 'combined', 'answer'], 'Group order differs.')
    require(torch.equal(vector['pair_mean_sums'][emotion], state['difference_sum'].float()), 'Vector sum differs from checkpoint.')
    require(torch.equal(vector['pair_counts'][emotion], torch.full((32,), len(ids))), 'Pair counts differ.')
    for layer in range(32):
        loaded, _ = vanilla.task.winners.read_vector(folder/'style-vectors.pt', emotion, layer, 'answer')
        expected = state['difference_sum'].float()[layer, 3]/len(ids)
        require(torch.equal(loaded, expected), f'L{layer}: loaded direction differs from sum/count.')
        require(torch.isfinite(loaded).all().item(), f'L{layer}: nonfinite direction.')
    natural = {}
    for row in read_rows(root/'delivery/results.jsonl'):
        key = row['question_id'], row['condition']
        require(key not in natural, f'Duplicate natural response: {key}')
        natural[key] = row
    compared = 0
    for saved in state['rows']:
        for style, field in [('neutral', 'neutral_response'), (emotion, 'emotion_response')]:
            row = natural.get((saved['question_id'], style))
            if row is None:
                continue
            require(row['response'] == saved[field], f'Collection/natural response mismatch: {saved["question_id"]}/{style}')
            require(row['prompt'] == plan['prompt'], 'Natural prompt differs.')
            require(row.get('strength') == 0. and row.get('interventions') == [], 'Natural answer has steering metadata.')
            compared += 1
    rows = state['rows']
    n = [r['neutral_forwards'] for r in rows]
    e = [r['emotion_forwards'] for r in rows]
    m = [r['matched_forwards'] for r in rows]
    require(all(v == min(a, b) and v > 0 for a, b, v in zip(n, e, m)), 'Matched forward counts differ from min(neutral, emotion).')
    retained = [100*v/b for v, b in zip(m, e)]
    print(f'{emotion}/{regime}: PASS config, {len(ids)} fitting IDs, sign metadata, all 32 loaded layer vectors, and single sum/count division.')
    print(f'Collection vs natural text: {compared}/{2*len(ids)} available responses identical.')
    print(f'Forwards mean/median: neutral={mean(n):.1f}/{median(n):g}; emotion={mean(e):.1f}/{median(e):g}; matched={mean(m):.1f}/{median(m):g}.')
    print(f'Emotion forwards retained per pair: mean={mean(retained):.1f}%; median={median(retained):.1f}%; '
          f'truncated in {sum(b>a for a,b in zip(n,e))}/{len(ids)} pairs.')
    job_path = folder/'strength-1-unfitted/run_config.json'
    if job_path.exists():
        job = json.loads(job_path.read_text())
        require(job['source'].get('emotion') == emotion, 'Steering loads another emotion.')
        require(job['source']['vector_sha256'] == vanilla.task.winners.digest(folder/'style-vectors.pt'), 'Steering vector hash differs.')
        require(job['target']['prompt'] == plan['prompt'], 'Steering prompt differs.')
        require(job['target']['max_new_tokens'] == plan['max_new_tokens'], 'Steering budget differs.')
        require(job['layer'] == 31 and job['group'] == 'answer' and job['strengths'] == [0.,1.], 'Steering layer/group/strength differs.')
        targets = {r['question_id'] for r in job['target']['records']}
        require(not targets.intersection(ids), 'Fitting/evaluation overlap.')
        require(targets == set(plan['bank'])-set(ids), 'Steering does not cover exact unfitted cohort.')
        checked = vanilla.steering_rows(job, job_path.parent)
        baseline = 0
        steered = 0
        for row in checked.values():
            if row['strength'] == 0.:
                old = natural.get((row['question_id'], 'neutral'))
                require(old is not None and row['response'] == old['response'], 'Steering baseline differs from natural neutral.')
                baseline += 1
            else:
                require(row.get('steered_forwards', 0) > 0, 'Saved intervention has zero/missing hook executions.')
                steered += 1
        print(f'L31 saved steering: PASS exact emotion/vector/prompt/audio metadata; {baseline} identical baselines; {steered} nonzero-hook responses.')
    else:
        print('L31 steering not prepared; collection checks only.')
    drift = folder/'layer-drift.csv'
    if drift.exists():
        print('Layer  ||applied mean direction||  Mean unit-direction consistency')
        with drift.open() as stream:
            for row in csv.DictReader(stream):
                if row['group'] == 'answer':
                    layer = int(row['layer'])
                    direction, _ = vanilla.task.winners.read_vector(folder/'style-vectors.pt', emotion, layer, 'answer')
                    print(f'{layer:5} {float(direction.norm()):27.4f} {float(row["shift_consistency"]):32.4f}')
    print('Pooling caveat: every forward averages all 512 answer slots, including future masked slots. '
          'Matched forwards are chronological prefixes, not matched reasoning stages. '
          'Direction consistency measures agreement between pair shifts, not cosine with reasoning.')
    return dict(pairs=len(ids), compared_responses=compared, mean_emotion_forward_retained_pct=mean(retained))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/diffa-gsm-vanilla'))
    parser.add_argument('--emotion', choices=vanilla.EMOTIONS, default='angry')
    parser.add_argument('--regimes', choices=vanilla.REGIMES, nargs='+', default=list(vanilla.REGIMES))
    args = parser.parse_args()
    print('CPU-only saved-data audit; no model load, generation, or file changes.')
    for regime in args.regimes:
        audit(args.root, args.emotion, regime)


if __name__ == '__main__':
    main()
