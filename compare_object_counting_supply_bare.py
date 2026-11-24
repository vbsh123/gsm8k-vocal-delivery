"""Compare saved neutral/angry Supply reasoning answers with NEUTRAL bare answers. CPU only."""
import argparse
import json
from pathlib import Path
from statistics import mean,median

import run_object_counting_supply_reasoning as run
from rescore_audio_steering import HarnessTaskScorer,paired


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=Path('outputs/diffa-object-counting-prompts'))
    parser.add_argument('--run',type=Path,default=Path('outputs/diffa-object-counting-supply-reasoning'))
    args=parser.parse_args()
    config=args.run/'run_config.json'
    if not config.exists():
        print('Supply reasoning condition not prepared yet.');return
    plan=json.loads(config.read_text())
    if plan['protocol']!=run.PROTOCOL or plan['prompts']!={c:run.PROMPT for c in run.CONDITIONS}:
        raise ValueError('Need the approved Supply reasoning conditions.')
    old=run.parent_plan(args.parent,plan)
    rows=[r for r in run.base.raw_rows(args.parent,old).values() if r['condition']=='neutral']
    rows+=list(run.base.raw_rows(args.run,plan).values())
    # Keep scorer receipts separate from the active runner's files.
    scorer=HarnessTaskScorer('bbh_cot_zeroshot_object_counting',args.run/'compare-neutral-bare-scoring')
    docs=[dict(input=plan['bank'][r['question_id']]['original_input'],target=plan['bank'][r['question_id']]['original_target']) for r in rows]
    scores=scorer.score([r['response'] for r in rows],docs)
    groups={c:{} for c in ('neutral',*run.CONDITIONS)}
    for i,row in enumerate(rows):
        groups[row['condition']][row['question_id']]=dict(row,metrics={m:v[i]['correct'] for m,v in scores.items()})
    found=False
    for condition in run.CONDITIONS:
        ids=sorted(groups['neutral'].keys()&groups[condition].keys())
        if not ids:continue
        found=True
        print(f'\n{condition} vs NEUTRAL BARE: SAME N={len(ids)}')
        for c,label in (('neutral','neutral-bare'),(condition,condition)):
            tokens=[groups[c][q]['response_tokens'] for q in ids]
            print(f'  {label} mean/median tokens: {mean(tokens):.1f}/{median(tokens):g}')
        for metric in scores:
            stats=paired([(groups['neutral'][q]['metrics'][metric],groups[condition][q]['metrics'][metric]) for q in ids])
            n=len(ids);b=stats['baseline_correct'];s=stats['steered_correct']
            print(f'  {metric}: {b}/{n} ({b/n:.1%}) -> {s}/{n} ({s/n:.1%}); '
                  f'gain={stats["gain_pp"]:+.2f} pp; p={stats["mcnemar_exact_p"]:.4g}; CI={stats["ci95_gain_pp"]}')
    if not found:print('No completed Supply reasoning answers matched to neutral-bare yet.')
    print('\nSaved responses only; exact paired McNemar p and paired bootstrap95% CI, unadjusted.')


if __name__=='__main__':main()
