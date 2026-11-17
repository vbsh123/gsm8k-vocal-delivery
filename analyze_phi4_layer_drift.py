"""CPU-only layer drift from the existing Phi SAME100 initial-forward states.

Ranks representation change, NOT causal importance or steering performance.
"""
import argparse
import csv
import json
from pathlib import Path

from collect_instruction_controls import validate
from audit_reasoning_emotion import PROTOCOLS


def analyze(source, output, top=8):
    import torch
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    plan = json.loads((source/'run_config.json').read_text())
    if plan['protocol'] != PROTOCOLS['phi4']:
        raise ValueError('Expected existing Phi SAME100 collection, not L17-only generation means.')
    neutral, angry, ids = [], [], []
    for i, pair in enumerate(plan['pairs']):
        paths = [source/'states'/f'{i:03d}-{c}.pt' for c in ('neutral-reasoning', 'angry-reasoning')]
        if not all(p.exists() for p in paths): continue
        hs = [validate(torch.load(path, map_location='cpu', weights_only=True), plan, pair, condition,
                       plan['reasoning_prompt'], plan['protocol'], 'phi4')
              for path, condition in zip(paths, ('neutral-reasoning', 'angry-reasoning'))]
        neutral.append(hs[0]); angry.append(hs[1]); ids.append(pair['question_id'])
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError('Need all100 unique original matched question captures.')
    n, a = torch.stack(neutral), torch.stack(angry)
    delta = a-n; nn = n.norm(dim=-1); an = a.norm(dim=-1); dn = delta.norm(dim=-1)
    valid = (nn > 0)&(an > 0)
    cosine = (n*a).sum(-1)/(nn*an).clamp_min(1e-30)
    relative = 100*dn/nn.clamp_min(1e-30)
    units = delta/dn.clamp_min(1e-30).unsqueeze(-1)
    summaries, pairs = [], []
    for g, group in enumerate(('answer', 'audio')):
        for layer in range(32):
            keep = valid[:, layer, g]
            rel, cos = relative[:, layer, g][keep], cosine[:, layer, g][keep]
            if not keep.any(): raise ValueError('Layer/group has no nonzero paired representation norms.')
            count = int(keep.sum())
            # quantile(.5), unlike torch.median, averages the middle two entries for even N.
            summaries.append(dict(group=group, layer=layer, pairs=count,
                mean_relative_change_pct=float(rel.mean()), median_relative_change_pct=float(rel.quantile(.5)),
                mean_representation_cosine=float(cos.mean()), mean_shift_norm=float(dn[:, layer, g][keep].mean()),
                norm_of_mean_shift=float(delta[:, layer, g][keep].mean(0).norm()),
                shift_direction_consistency=float(units[:, layer, g][keep].mean(0).norm())))
            for i, qid in enumerate(ids):
                pairs.append(dict(question_id=qid, group=group, layer=layer, valid_nonzero_norms=bool(valid[i, layer, g]),
                    relative_change_pct=float(relative[i, layer, g]) if valid[i, layer, g] else None,
                    representation_cosine=float(cosine[i, layer, g]) if valid[i, layer, g] else None,
                    shift_norm=float(dn[i, layer, g])))
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('layer-summary.csv', summaries), ('pair-metrics.csv', pairs)]:
        with (output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    note = ('Initial native forward, not whole-generation pooled states. Same100 neutral/angry audio and reasoning prompt. '
            'Relative drift is ||angry-neutral||/||neutral|| per question, then mean across questions. '
            'Large change does not establish a useful steering layer or explain accuracy. '
            'Different audio lengths/positions can contribute. No outcome filtering or new inference.')
    (output/'summary.json').write_text(json.dumps(dict(protocol='phi4-same100-initial-layer-drift-v1',
        source=str(source.absolute()), questions=ids, rows=summaries, note=note), indent=2)+'\n')
    print('Phi SAME100: angry versus neutral; initial forward; ranked by mean relative activation change.')
    for group in ('answer', 'audio'):
        print(f'\n{group.upper()} positions:')
        print('Layer  Mean change %  Median change %  Mean cosine  Shift consistency')
        ranked = sorted((r for r in summaries if r['group'] == group), key=lambda r: r['mean_relative_change_pct'], reverse=True)
        for row in ranked[:top]:
            print(f'{row["layer"]:5d} {row["mean_relative_change_pct"]:14.2f} {row["median_relative_change_pct"]:16.2f}'
                  f' {row["mean_representation_cosine"]:12.4f} {row["shift_direction_consistency"]:18.4f}')
    print('\nFull32-layer tables:', output)
    print('Descriptive representation drift; not a ranking of proven steering effects.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('outputs/phi4-reasoning-emotion100-gsm8k'))
    p.add_argument('--output', type=Path, default=Path('outputs/phi4-gsm100-layer-drift'))
    p.add_argument('--top', type=int, default=8)
    args = p.parse_args()
    if not 1 <= args.top <= 32: p.error('--top must be between1 and32')
    analyze(args.source, args.output, args.top)


if __name__ == '__main__': main()
