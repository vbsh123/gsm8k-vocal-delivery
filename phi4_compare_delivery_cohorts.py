"""Compare old and fresh Phi neutral/angry answers on identical held-out questions; CPU scoring only."""
import argparse
import json
from pathlib import Path

from steer import read_rows


def matched_deliveries(rows, *, heldout=False):
    found = {}
    for row in rows:
        style = row.get('condition',row.get('style'))
        if (style not in ('neutral','angry') or row.get('interventions')
                or float(row.get('strength') or 0.) != 0.
                or heldout and row.get('split') != 'test'):
            continue
        key = row['question_id'],row['voice'],style
        if key in found and any(found[key].get(k) != row.get(k) for k in ('response','audio_sha256','prompt')):
            raise ValueError(f'Conflicting saved answers for {key}')
        found[key] = dict(row,condition=style)
    ids = set((q,v) for q,v,s in found if s=='neutral') & set((q,v) for q,v,s in found if s=='angry')
    return {key:row for key,row in found.items() if key[:2] in ids}


def compare(old, new, score, paired):
    old_ids, new_ids = {k[:2] for k in old}, {k[:2] for k in new}
    common = sorted(old_ids & new_ids)
    if not common:
        raise ValueError('No common questions with both saved deliveries; no inference started.')
    output = dict(old_complete_questions=len(old_ids),fresh_heldout_complete_questions=len(new_ids),
                  common_question_voice_ids=common,old_questions_missing_fresh_heldout=len(old_ids-new_ids),
                  cohorts={},cross_run={})
    scored = {}
    for label,records,ids in [('old-full',old,sorted(old_ids)),('old-common',old,common),('fresh-common',new,common)]:
        rows = [records[q,v,s] for q,v in ids for s in ('neutral','angry')]
        values = score(rows)
        metrics = {(r['question_id'],r['voice'],r['condition']):r['metrics'] for r in values}
        scored[label] = metrics
        output['cohorts'][label] = {metric:paired([(metrics[q,v,'neutral'][metric],metrics[q,v,'angry'][metric]) for q,v in ids])
                                   for metric in values[0]['metrics']}
    for style in ('neutral','angry'):
        output['cross_run'][style] = dict(
            same_audio=sum(old[q,v,style].get('audio_sha256') == new[q,v,style].get('audio_sha256') for q,v in common),
            same_prompt=sum(old[q,v,style].get('prompt') == new[q,v,style].get('prompt') for q,v in common),
            same_response=sum(old[q,v,style]['response'] == new[q,v,style]['response'] for q,v in common),
            same_budget=sum(old[q,v,style].get('max_new_tokens') == new[q,v,style].get('max_new_tokens') for q,v in common),
            paired={metric:paired([(scored['old-common'][q,v,style][metric],scored['fresh-common'][q,v,style][metric]) for q,v in common])
                    for metric in scored['old-common'][common[0][0],common[0][1],style]})
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-run',type=Path,default=Path('outputs/phi4-gsm400-all-layers-test869-minus1'))
    parser.add_argument('--fresh-run',type=Path,default=Path('outputs/phi4-diffa-cohorts-three-directions/gsm8k'))
    parser.add_argument('--output',type=Path,default=Path('outputs/phi4-gsm-cohort-comparison'))
    args=parser.parse_args()
    old_plan=json.loads((args.old_run/'run_config.json').read_text())
    fresh_plan=json.loads((args.fresh_run/'run_config.json').read_text())
    paths=sorted(args.old_run.glob('references*.jsonl')) or [args.old_run/'results.jsonl']
    old=matched_deliveries([r for path in paths for r in read_rows(path)])
    new=matched_deliveries(read_rows(args.fresh_run/'deliveries.jsonl'),heldout=True)
    for (q,v,s) in set(old) & set(new):
        if str(old[q,v,s]['reference_answer']).strip() != str(new[q,v,s]['reference_answer']).strip():
            raise ValueError(f'Gold answer changed for {q}; inspect before comparing.')
    from winning_diffa_sweeps import score_rows,atomic_text
    from rescore_audio_steering import paired
    args.output.mkdir(parents=True,exist_ok=True)
    def score(rows):
        return score_rows(dict(target=dict(task='gsm8k',name='gsm8k',bank=fresh_plan['bank'])),rows,args.output)[0]
    result=compare(old,new,score,paired)
    result.update(old_run=str(args.old_run),fresh_run=str(args.fresh_run),
                  old_prompt=old_plan['prompt'],fresh_prompt=fresh_plan['prompt'],
                  note='Same common question/voice IDs, no correctness or emotion filtering. Collection answers excluded. '
                       'Audio/prompt/budget changes are reported; same question IDs alone do not isolate every other change.')
    for label,metrics in result['cohorts'].items():
        print('\n'+label+' — angry versus neutral')
        for metric,r in metrics.items():
            print(f'{metric:18} N={r["pairs"]}; neutral={r["baseline_correct"]}/{r["pairs"]}, angry={r["steered_correct"]}/{r["pairs"]}; '
                  f'gain={r["gain_pp"]:+.2f} pp; p={r["mcnemar_exact_p"]:.4g}; CI={r["ci95_gain_pp"]}')
    n=len(result['common_question_voice_ids'])
    print(f'\nCommon held-out pairs: {n}; old cohort questions without fresh held-out answers: {result["old_questions_missing_fresh_heldout"]}.')
    for style,checks in result['cross_run'].items():
        print(f'{style}: same audio {checks["same_audio"]}/{n}; same prompt {checks["same_prompt"]}/{n}; '
              f'same token budget {checks["same_budget"]}/{n}; identical response {checks["same_response"]}/{n}.')
    path=args.output/'summary.json'
    atomic_text(path,json.dumps(result,indent=2)+'\n')
    print('Saved:',path,'— CPU scoring only; no generations or changes to the running queue.')


if __name__=='__main__':
    main()
