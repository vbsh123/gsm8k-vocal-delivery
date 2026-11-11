"""CPU-only drift analysis of saved neutral/angry activation means; no model loading.

Reads collection-config.json and pairs/NNNNN.pt from extract_audio_vectors.py
or the older extract_jenny_svd_probe.py. Each tensor is [layer, group, hidden].
These are token/forward averages, not aligned token-level diffusion states.
"""

import argparse
from collections import Counter, defaultdict
import csv
from html import escape
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

from scoring import score_response, SCORER_VERSION


GROUPS=['acoustic','semantic','combined','answer']
OUTCOMES=['recovery','regression','both_correct','both_wrong']


def outcome(neutral_correct, angry_correct):
    if angry_correct and not neutral_correct:
        return 'recovery'
    if neutral_correct and not angry_correct:
        return 'regression'
    return 'both_correct' if neutral_correct else 'both_wrong'


def number(value):
    value=float(value)
    return value if math.isfinite(value) else None


def average(values):
    values=[v for v in values if v is not None]
    return mean(values) if values else None


def vector_metrics(neutral, angry, torch):
    neutral,angry=neutral.double(),angry.double()
    delta=angry-neutral
    nn=neutral.norm(dim=-1)
    an=angry.norm(dim=-1)
    dn=delta.norm(dim=-1)
    cosine=(neutral*angry).sum(-1)/(nn*an).clamp_min(1e-30)
    cosine=torch.where((nn>0)&(an>0),cosine.clamp(-1,1),float('nan'))
    relative=torch.where(nn>0,100*dn/nn,float('nan'))
    units=delta/dn.clamp_min(1e-30)[...,None]
    return delta,units,dn>0,cosine,relative,dn


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display(value, digits=2):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def html_report(metadata, summaries, comparisons, output):
    by_key={(r['layer'],r['group'],r['outcome']):r for r in summaries}
    sections=[]
    for group in metadata['groups']:
        rows=[]
        for comparison in [r for r in comparisons if r['group']==group]:
            layer=comparison['layer']
            cells=[]
            for bucket in OUTCOMES:
                value=by_key[layer,group,bucket]['mean_relative_l2_pct']
                cells.append(f'<td>{display(value)}%</td>' if value is not None else '<td>n/a</td>')
            rows.append(f'<tr><td>{layer}</td>'+''.join(cells)+
                f'<td>{display(comparison["recovery_minus_regression_drift_pp"])}</td>'+
                f'<td>{display(comparison["cosine_mean_recovery_vs_regression_shift"],3)}</td>'+
                f'<td>{display(comparison["recovery_coherence"],3)}</td>'+
                f'<td>{display(comparison["regression_coherence"],3)}</td></tr>')
        sections.append(f'<h2>{escape(group)}</h2><table><thead><tr><th>Layer</th>'
            '<th>Recovery<br>drift %</th><th>Regression<br>drift %</th><th>Both correct<br>drift %</th>'
            '<th>Both wrong<br>drift %</th><th>Recovery − regression<br>drift points</th>'
            '<th>Cosine between<br>mean shift directions</th><th>Recovery<br>coherence</th><th>Regression<br>coherence</th>'
            '</tr></thead><tbody>'+''.join(rows)+'</tbody></table>')
    counts='; '.join(f'{name}: {metadata["outcome_counts"].get(name,0)}' for name in OUTCOMES)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Paired audio representation drift</title>'
        '<style>body{font:15px system-ui;max-width:1250px;margin:30px auto;padding:0 20px;line-height:1.5}'
        'table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}'
        'th{background:#edf2f7}th:first-child,td:first-child{text-align:left}tbody tr:nth-child(even){background:#f8fafc}'
        'h2{margin-top:35px}p{max-width:1000px}</style><h1>Neutral versus angry: saved representation means</h1>'
        f'<p><b>{metadata["completed_pairs"]}/{metadata["expected_pairs"]} saved pairs; {metadata["unique_questions"]} unique questions.</b> '
        f'{escape(counts)}.</p><p>Source: {escape(metadata["collection"])}; averaging: {escape(metadata["averaging"])}. '
        'Outcome labels come from the unsteered neutral/angry answers saved during collection. Recovery does not mean steering recovery.</p>'
        '<p>Drift % = 100 × ‖angry − neutral‖ / ‖neutral‖, calculated separately for each question, layer and group, then averaged. '
        'These are drifts of saved token/forward-pooled means, not averages of per-token drift magnitudes. '
        'Layers are zero-based. Combined means acoustic + semantic audio positions; answer is a separate group.</p>'
        '<p>Coherence = length of the average unit shift direction (0–1). Higher means shifts point more consistently in one direction. '
        'The direction cosine compares mean recovery and regression shifts. Neither measure establishes which layer controls accuracy. '
        'Matched forward counts do not establish matched denoising states; generated answers can already differ.</p>'
        '<p><a href="layer-summary.csv">All layer statistics, including cosine similarity</a> · '
        '<a href="pair-metrics.csv">Per-question metrics</a> · <a href="outcome-comparison.csv">Outcome comparisons</a> · '
        '<a href="summary.json">Collection metadata and limitations</a></p>'+''.join(sections))


def analyze(collection, output):
    import torch
    torch.set_num_threads(min(8,torch.get_num_threads()))
    config_path=collection/'collection-config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'Missing {config_path}. Use the collection with pairs/NNNNN.pt containing neutral and angry means; no audio/model files are needed.')
    plan=json.loads(config_path.read_text())
    groups=plan.get('groups',GROUPS)
    averaging=plan.get('averaging','matched')
    output.mkdir(parents=True,exist_ok=True)
    buckets=defaultdict(list)
    counts=Counter()
    forwards=defaultdict(list)
    questions=set()
    shifts={}
    unit_sums={}
    direction_counts={}
    shape=None
    completed=0
    missing=[]
    with (output/'pair-metrics.csv').open('w',newline='') as stream,torch.inference_mode():
        writer=None
        for index,pair in enumerate(plan['pairs']):
            path=collection/'pairs'/f'{index:05d}.pt'
            if not path.exists():
                missing.append(index)
                continue
            row=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            if 'neutral' not in row or 'angry' not in row:
                raise ValueError(f'{path} contains differences only; use the original saved neutral/angry means, not temporal-vector aggregates.')
            if any(row['pair'][key]!=pair[key] for key in ('question_id','neutral_id','angry_id')):
                raise ValueError(f'Pair identity differs from collection config: {path}')
            if row.get('averaging',averaging)!=averaging:
                raise ValueError(f'Mixed averaging methods in {path}')
            n,a=row['neutral'],row['angry']
            if n.ndim!=3 or n.shape!=a.shape or n.shape[1]!=len(groups):
                raise ValueError(f'Expected matching [layer,group,hidden] means: {path}')
            if not torch.isfinite(n).all() or not torch.isfinite(a).all():
                raise ValueError(f'Nonfinite activation means: {path}')
            if shape is None:
                shape=tuple(n.shape)
                for bucket in ['all',*OUTCOMES]:
                    shifts[bucket]=torch.zeros(shape,dtype=torch.float64)
                    unit_sums[bucket]=torch.zeros(shape,dtype=torch.float64)
                    direction_counts[bucket]=torch.zeros(shape[:2],dtype=torch.long)
            elif tuple(n.shape)!=shape:
                raise ValueError(f'Inconsistent activation shape: {path}')
            reference=plan['manifest'][pair['neutral_id']]['answer']
            other_reference=plan['manifest'][pair['angry_id']]['answer']
            if str(reference)!=str(other_reference):
                raise ValueError(f'Mismatched reference answers: {pair["question_id"]}')
            nc=score_response(row['neutral_response'],reference)['correct']
            ac=score_response(row['angry_response'],reference)['correct']
            label=outcome(nc,ac)
            delta,units,valid,cosine,relative,norm=vector_metrics(n,a,torch)
            for bucket in ['all',label]:
                shifts[bucket]+=delta
                unit_sums[bucket]+=units
                direction_counts[bucket]+=valid
                counts[bucket]+=1
            questions.add(pair['question_id'])
            for field in ('neutral_forwards','angry_forwards','matched_forwards'):
                if row.get(field) is not None:
                    forwards[label,field].append(row[field])
            for layer in range(shape[0]):
                for g,group in enumerate(groups):
                    c=number(cosine[layer,g]); r=number(relative[layer,g]); magnitude=number(norm[layer,g])
                    metric=dict(pair_index=index,question_id=pair['question_id'],speaker=pair.get('speaker'),outcome=label,
                                neutral_correct=nc,angry_correct=ac,layer=layer,group=group,
                                cosine_similarity=c,relative_l2_pct=r,shift_norm=magnitude,averaging=averaging,
                                neutral_forwards=row.get('neutral_forwards'),angry_forwards=row.get('angry_forwards'),
                                matched_forwards=row.get('matched_forwards'))
                    if writer is None:
                        writer=csv.DictWriter(stream,fieldnames=list(metric));writer.writeheader()
                    writer.writerow(metric)
                    for bucket in ['all',label]:
                        buckets[layer,group,bucket].append((r,c,magnitude))
            completed+=1
            if completed%50==0:
                print(f'CPU analysis: {completed}/{len(plan["pairs"])} pairs read.',flush=True)
    if shape is None:
        raise ValueError('No saved pair means found. This script does not collect activations or run a model.')
    summaries=[]
    for layer in range(shape[0]):
        for g,group in enumerate(groups):
            for bucket in ['all',*OUTCOMES]:
                values=buckets[layer,group,bucket]
                rel=[r for r,c,d in values if r is not None]
                count=int(direction_counts[bucket][layer,g])
                length=float(unit_sums[bucket][layer,g].norm())
                summaries.append(dict(layer=layer,group=group,outcome=bucket,pairs=len(values),relative_pairs=len(rel),nonzero_shift_pairs=count,
                    mean_relative_l2_pct=average(rel),median_relative_l2_pct=median(rel) if rel else None,
                    mean_cosine_similarity=average([c for r,c,d in values]),mean_shift_norm=average([d for r,c,d in values]),
                    direction_coherence=min(1.,length/count) if count else None,
                    mean_pairwise_shift_cosine=max(-1.,min(1.,(length**2-count)/(count*(count-1)))) if count>1 else None))
    lookup={(r['layer'],r['group'],r['outcome']):r for r in summaries}
    comparisons=[]
    for layer in range(shape[0]):
        for g,group in enumerate(groups):
            rec=lookup[layer,group,'recovery']; reg=lookup[layer,group,'regression']
            rm,gm=rec['mean_relative_l2_pct'],reg['mean_relative_l2_pct']
            x,y=shifts['recovery'][layer,g],shifts['regression'][layer,g]
            denominator=float(x.norm()*y.norm())
            comparisons.append(dict(layer=layer,group=group,recovery_pairs=counts['recovery'],regression_pairs=counts['regression'],
                recovery_mean_drift_pct=rm,regression_mean_drift_pct=gm,
                recovery_minus_regression_drift_pp=rm-gm if rm is not None and gm is not None else None,
                cosine_mean_recovery_vs_regression_shift=max(-1.,min(1.,float(x.dot(y))/denominator)) if denominator else None,
                recovery_coherence=rec['direction_coherence'],regression_coherence=reg['direction_coherence']))
    metadata=dict(collection=str(collection.resolve()),expected_pairs=len(plan['pairs']),completed_pairs=completed,
                  source_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  missing_pair_indices=missing,unique_questions=len(questions),shape=list(shape),groups=groups,averaging=averaging,
                  seed=plan.get('seed'),scorer_version=SCORER_VERSION,outcome_counts={k:counts[k] for k in OUTCOMES},
                  neutral_correct=counts['both_correct']+counts['regression'],angry_correct=counts['both_correct']+counts['recovery'],
                  forward_counts={bucket:{field:average(forwards[bucket,field]) for field in ('neutral_forwards','angry_forwards','matched_forwards')} for bucket in OUTCOMES},
                  limitations=['Token/forward means only; no individual token or step states.',
                               'Matched forward indices are not guaranteed to be matched diffusion states.',
                               'Outcome groups describe natural angry versus neutral answers, not steering effects.',
                               'The extraction cohort was filtered for classifier agreement; it is not the unfiltered benchmark.',
                               'Descriptive layer comparisons do not establish causal control.'])
    write_csv(output/'layer-summary.csv',summaries)
    write_csv(output/'outcome-comparison.csv',comparisons)
    (output/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
    html_report(metadata,summaries,comparisons,output)
    print(f'\nAnalyzed {completed}/{len(plan["pairs"])} pairs on CPU; outcomes: {metadata["outcome_counts"]}')
    for group in groups:
        ranked=sorted((r for r in comparisons if r['group']==group and r['recovery_minus_regression_drift_pp'] is not None),
                      key=lambda r:abs(r['recovery_minus_regression_drift_pp']),reverse=True)[:3]
        print(f'\n{group}: largest recovery/regression drift contrasts (descriptive)')
        for row in ranked:
            print(f'  L{row["layer"]}: recovery={display(row["recovery_mean_drift_pct"])}%, regression={display(row["regression_mean_drift_pct"])}%, shift-direction cosine={display(row["cosine_mean_recovery_vs_regression_shift"],3)}')
    print(f'Open {output}/index.html. No DIFFA model or GPU was loaded.',flush=True)
    return metadata,summaries,comparisons


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,default=Path('outputs/jenny-svd-probe'))
    p.add_argument('--output',type=Path,default=Path('outputs/jenny-drift-analysis'))
    args=p.parse_args()
    analyze(args.collection,args.output)


if __name__=='__main__':
    main()
