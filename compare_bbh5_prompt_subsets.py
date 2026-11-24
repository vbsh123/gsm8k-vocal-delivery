"""Compare saved BBH5 natural deliveries on the full250 and original fixed125 subsets. CPU only."""
import argparse
from collections import defaultdict
import hashlib
import json
from math import comb
from pathlib import Path

METRICS=('flexible-extract','strict-match')
STYLES=('neutral','angry','happy','sad')


def read_rows(path):
    lines=path.read_bytes().splitlines(keepends=True)
    result=[]
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:result.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):
            if i==len(lines)-1 and not line.endswith(b'\n'):break
            raise
    return result


def paired(a,b):
    import numpy as np
    n=len(a)
    if not n:return None
    w=sum(not x and y for x,y in zip(a,b));l=sum(x and not y for x,y in zip(a,b));d=w+l
    p=min(1.,2*sum(comb(d,i) for i in range(min(w,l)+1))/2**d)
    draws=np.random.default_rng(42).multinomial(n,[w/n,l/n,1-(w+l)/n],size=200000)
    return dict(n=n,baseline_correct=sum(a),test_correct=sum(b),gain_pp=100*(sum(b)-sum(a))/n,
                wins=w,losses=l,p=p,ci95=np.percentile(100*(draws[:,0]-draws[:,1])/n,[2.5,97.5]).tolist())


def index(rows):
    groups=defaultdict(dict)
    for row in rows:
        s=row['condition'];q=row['question_id']
        if s not in STYLES:raise ValueError('Expected natural deliveries only.')
        if q in groups[s]:raise ValueError(f'Duplicate {s}/{q}; refusing to choose an answer.')
        if any(not isinstance(row['metrics'].get(m),bool) for m in METRICS):
            raise ValueError('Need saved benchmark strict/flexible boolean scores; no heuristic rescoring.')
        groups[s][q]=row
    return groups


def compare(reference,plan,rows):
    if reference['protocol']!='bbh5-original-natural-reference-v1' or plan.get('name')!='bbh5':
        raise ValueError('Need the original BBH5 reference and current BBH5 natural run.')
    old=index(reference['rows']);new=index(rows)
    all_ids=set(reference['question_ids']);held=set(reference['heldout_ids'])
    if len(all_ids)!=250 or len(held)!=125 or not held<=all_ids:
        raise ValueError('Reference must contain the exact original 250/125 IDs.')
    if set(plan['question_order'])!=all_ids:
        raise ValueError('Current full question pool differs from the original 250 IDs.')
    for key,value in reference['recipe'].items():
        if plan.get('recipe',{}).get(key,plan.get(key))!=value:
            raise ValueError(f'Decoding setting changed: {key}.')
    if plan['prompt']==reference['prompt']:
        raise ValueError('Current and original prompts are identical; this is not the bare-versus-original comparison.')
    expected={(r['question_id'],r['condition']):r for r in plan['records']}
    for s,group in new.items():
        for q,row in group.items():
            record=expected.get((q,s))
            if record is None or row.get('audio_sha256')!=record['audio_sha256'] or row.get('prompt')!=plan['prompt']:
                raise ValueError(f'Current saved answer differs from its configured input: {q}/{s}.')
            if row.get('interventions') or float(row.get('strength') or 0.)!=0.:
                raise ValueError('Intervened answer entered natural delivery comparison.')
            if s in ('neutral','angry'):
                previous=old[s].get(q)
                if previous is None or any(row.get(k)!=previous[k] for k in ('audio_sha256','voice','reference_answer')):
                    raise ValueError(f'Original/current audio or reference differs: {q}/{s}.')
    output=[]
    for scope,ids in [('all250',all_ids),('original-heldout125',held),('original-fitting125',all_ids-held)]:
        for version,groups in [('original',old),('current',new)]:
            for s in ('angry','happy','sad'):
                qs=sorted(ids&groups.get('neutral',{}).keys()&groups.get(s,{}).keys())
                if not qs:continue
                for metric in METRICS:
                    stats=paired([groups['neutral'][q]['metrics'][metric] for q in qs],
                                 [groups[s][q]['metrics'][metric] for q in qs])
                    output.append(dict(scope=scope,version=version,reference='neutral',emotion=s,metric=metric,expected=len(ids),**stats))
        # These test prompt changes for the same condition on the same IDs.
        for s in ('neutral','angry'):
            qs=sorted(ids&old[s].keys()&new.get(s,{}).keys())
            if not qs:continue
            for metric in METRICS:
                stats=paired([old[s][q]['metrics'][metric] for q in qs],[new[s][q]['metrics'][metric] for q in qs])
                output.append(dict(scope=scope,version='current-minus-original',reference='original prompt',emotion=s,
                                   metric=metric,expected=len(ids),**stats))
    return output


def report(results):
    for scope in ('all250','original-heldout125','original-fitting125'):
        print(f'\n{scope}: fixed original IDs; no emotion-classifier or accuracy filtering.')
        for r in results:
            if r['scope']!=scope:continue
            label=f'{r["version"]}: {r["emotion"]} vs {r["reference"]}'
            print(f'{label:58} {r["metric"]:18} N={r["n"]}/{r["expected"]}; '
                  f'{r["baseline_correct"]}/{r["n"]} ({r["baseline_correct"]/r["n"]:.1%}) -> '
                  f'{r["test_correct"]}/{r["n"]} ({r["test_correct"]/r["n"]:.1%}); '
                  f'gain={r["gain_pp"]:+.2f} pp; p={r["p"]:.4g}; CI={r["ci95"]}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current',type=Path,default=Path('outputs/emotion-block-night/diffa/bbh5'))
    parser.add_argument('--reference',type=Path,default=Path(__file__).parent/'data/bbh5_original_natural_reference.json')
    parser.add_argument('--output',type=Path,default=Path('outputs/bbh5-original-subset-comparison.json'))
    args=parser.parse_args()
    plan_path=args.current/'run_config.json';scores_path=args.current/'delivery-rescored.jsonl'
    if not scores_path.exists():
        raise FileNotFoundError('First run: bash run_emotion_block_night.sh new --mode summary (CPU scoring only).')
    print('CPU-only saved-score comparison. No generation, collection, steering, model loading or input-file changes.')
    reference=json.loads(args.reference.read_text());plan=json.loads(plan_path.read_text())
    rows=read_rows(scores_path)
    config_hash=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if any(r.get('run_config_sha256')!=config_hash for r in rows):
        raise ValueError('Saved scores are not bound to the current frozen configuration; refresh the CPU summary.')
    summary_path=args.current/'delivery-summary.json'
    current_scorer=json.loads(summary_path.read_text())['scorer']
    if current_scorer.get('task')!=reference['scorer'].get('task'):
        raise ValueError('Original and current benchmark scoring tasks differ.')
    for filename,checksum in reference['scorer'].get('files',{}).items():
        if filename.startswith('tasks/bbh/') and current_scorer.get('files',{}).get(filename)!=checksum:
            raise ValueError(f'Original and current BBH extraction configuration differs: {filename}.')
    results=compare(reference,plan,rows);report(results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    receipt=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(dict(results=results,original_prompt=reference['prompt'],current_prompt=plan['prompt'],
        sources=[receipt(args.reference),receipt(plan_path),receipt(scores_path),receipt(summary_path)],
        note='Exploratory unadjusted exact McNemar tests and 95% question-paired bootstrap intervals. '
             'All250 and fitting125 natural-audio comparisons include vector-fitting examples; no steering is analyzed. '
             'The earlier natural reference includes neutral and angry only; no old happy/sad responses are invented.'),indent=2)+'\n')
    print('Saved:',args.output)


if __name__=='__main__':main()
