"""Rescore saved answers, using upstream GSM8K extraction and independent choice parsing. No GPU."""

import argparse
import ast
import csv
from fractions import Fraction
import hashlib
import json
from math import comb
from pathlib import Path
import re
from urllib.request import Request, urlopen

from audit_commonsense_scoring import relaxed_choice,RELAXED_SCORER_VERSION
from evaluate_commonsense_audio import score_choice
from scoring import score_response
from steer import read_rows

HARNESS='https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/'
FLEXIBLE=r'(-?[$0-9.,]{2,})|(-?[0-9]+)'


def download(url):
    with urlopen(Request(url,headers={'User-Agent':'audio-steering-scoring-audit'}),timeout=120) as response:
        return response.read()


def harness_filter(cache):
    cache.mkdir(parents=True,exist_ok=True)
    revision=cache/'revision.txt'
    if not revision.exists():
        sha=json.loads(download('https://api.github.com/repos/EleutherAI/lm-evaluation-harness/commits/main'))['sha']
        revision.write_text(sha+'\n')
    sha=revision.read_text().strip()
    files={'extraction.py':'lm_eval/filters/extraction.py','gsm8k.yaml':'lm_eval/tasks/gsm8k/gsm8k.yaml'}
    for name,path in files.items():
        if not (cache/name).exists():(cache/name).write_bytes(download(HARNESS+sha+'/'+path))
    # Execute the upstream RegexFilter class unchanged, with only its registry/base dependencies removed.
    tree=ast.parse((cache/'extraction.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RegexFilter')
    cls.bases=[];cls.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
        ast.Import(names=[ast.alias(name='re')]),cls],type_ignores=[])
    namespace={};exec(compile(ast.fix_missing_locations(module),str(cache/'extraction.py'),'exec'),namespace)
    if FLEXIBLE not in (cache/'gsm8k.yaml').read_text():raise ValueError('Upstream GSM8K flexible filter changed; inspect before using')
    (cache/'provenance.json').write_text(json.dumps(dict(repository='EleutherAI/lm-evaluation-harness',revision=sha,
        files={name:hashlib.sha256((cache/name).read_bytes()).hexdigest() for name in files}),indent=2)+'\n')
    return namespace['RegexFilter']


def paired(values):
    import numpy as np
    n=len(values);b=sum(x for x,y in values);s=sum(y for x,y in values)
    w=sum(y and not x for x,y in values);l=sum(x and not y for x,y in values);d=w+l
    p=min(1.,2*sum(comb(d,i) for i in range(min(w,l)+1))/2**d)
    draws=np.random.default_rng(42).multinomial(n,[w/n,l/n,(n-d)/n],size=200000)
    ci=np.percentile(100*(draws[:,0]-draws[:,1])/n,[2.5,97.5]).tolist()
    return dict(pairs=n,baseline_correct=b,steered_correct=s,gain_pp=100*(s-b)/n,
        wins=w,losses=l,mcnemar_exact_p=p,ci95_gain_pp=ci)


def numeric_equal(a,b):
    try:return Fraction(a.replace(',','').replace('$','').strip())==Fraction(b.replace(',','').replace('$','').strip())
    except (ValueError,ZeroDivisionError):return False


def report(rows,task,bank,output,label,RegexFilter=None):
    output.mkdir(parents=True,exist_ok=True)
    latest={(r['question_id'],r.get('voice',r.get('speaker','')),r.get('condition','baseline')):r for r in rows}
    predictions={};disagreements=[]
    for key,row in latest.items():
        response=row['response'];gold=str(row['reference_answer'])
        if task=='commonsenseqa':
            if bank[key[0]]['answer']!=gold:raise ValueError('Saved reference differs from original CommonsenseQA source')
            clean=response.replace('**','').replace('__','').replace('`','')
            labels=re.findall(r'final\s+answer\s*:\s*([^\n]*)',clean,re.I)
            m=re.fullmatch(r'(?:option\s+)?\(?([A-E])\)?[.!]?',labels[-1].strip(),re.I) if labels else None
            strict=m[1].upper() if m else None
            relaxed=relaxed_choice(response,bank[key[0]]['choices'])
            scores={'current_strict':score_choice(response,gold)['correct'],
                'independent_strict':strict==gold,RELAXED_SCORER_VERSION:relaxed==gold}
            extracted=dict(independent_strict=strict,relaxed_choice=relaxed,relaxed_scorer_version=RELAXED_SCORER_VERSION)
        else:
            match=re.fullmatch(r'(?:spoken_)?gsm8k_test_(\d+)(?:__.*)?',key[0])
            if not match:raise ValueError(f'Unknown GSM8K source ID: {key[0]}')
            official=bank[int(match[1])]['answer'].rsplit('####',1)[-1].strip()
            if score_response('Final answer: '+gold,official)['correct'] is not True:raise ValueError('Saved GSM8K reference differs from original source')
            extracted=RegexFilter(regex_pattern=FLEXIBLE,group_select=-1).apply([[response]],[{}])[0][0]
            normalize=lambda s:re.sub(r'\.$','',s.replace(',','').replace('$','').strip()).lower()
            scores={'current_strict':score_response(response,gold)['correct'],
                'harness_flexible':normalize(extracted)==normalize(official),
                'harness_flexible_numeric':numeric_equal(extracted,official)}
        predictions[key]=scores
        if len(set(scores.values()))>1 or row.get('correct')!=scores['current_strict']:
            disagreements.append(dict(question_id=key[0],condition=key[2],gold=gold,response=response,
                stored_correct=row.get('correct'),scores=scores,extracted=extracted))
    print('\n'+label+' — completed matched pairs; parse failures count as incorrect')
    print(f'{"Scorer / condition":52} {"N":>5} {"Base":>7} {"Steered":>8} {"Gain pp":>8} {"W/L":>9} {"Exact p":>9}   95% CI gain pp')
    summaries=[]
    for condition in sorted({k[2] for k in latest}-{'baseline'}):
        keys=[k for k in latest if k[2]==condition and (*k[:2],'baseline') in latest]
        if not keys:continue
        for scorer in predictions[keys[0]]:
            result=paired([(predictions[(*k[:2],'baseline')][scorer],predictions[k][scorer]) for k in keys])
            n=result['pairs'];lo,hi=result['ci95_gain_pp']
            print(f'{scorer+" / "+condition:52} {n:5} {result["baseline_correct"]/n:7.1%} {result["steered_correct"]/n:8.1%} '
                f'{result["gain_pp"]:+8.2f} {str(result["wins"])+"/"+str(result["losses"]):>9} {result["mcnemar_exact_p"]:9.4f} [{lo:+.1f}, {hi:+.1f}]')
            summaries.append(dict(cohort=label,scorer=scorer,condition=condition,**result))
    (output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (output/'disagreements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in disagreements))
    with (output/'accuracy.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['condition','scorer','n','correct'])
        for condition in sorted({k[2] for k in latest}):
            keys=[k for k in latest if k[2]==condition]
            for scorer in predictions[keys[0]]:writer.writerow([condition,scorer,len(keys),sum(predictions[k][scorer] for k in keys)])
    print(f'Scoring disagreements: {len(disagreements)}; saved with full responses in {output}. Original results untouched.')
    return summaries


def commonsense_reports(original,extension,source,output):
    sourcefile=source/'commonsenseqa-validation-source.jsonl'
    if not sourcefile.exists():sourcefile=Path('outputs/azure-commonsenseqa500/commonsenseqa-validation-source.jsonl')
    from generate_commonsense_azure import questions,SOURCE_SHA256
    data=sourcefile.read_bytes()
    if hashlib.sha256(data).hexdigest()!=SOURCE_SHA256:raise ValueError('Original CommonsenseQA source checksum differs')
    bank={r['question_id']:r for r in questions(data,1221)}
    old=read_rows(original/'results.jsonl');new=read_rows(extension/'results.jsonl')
    split=json.loads((original/'split.json').read_text())
    train={r['question_id'] for r in split['extraction']}
    if train&{r['question_id'] for r in old+new}:raise ValueError('Extraction questions appeared in evaluation')
    if {r['question_id'] for r in old}&{r['question_id'] for r in new}:raise ValueError('Evaluation cohorts overlap')
    if not old and not new:raise ValueError('No saved CommonsenseQA steering answers')
    for label,rows in [('original-250',old),('new-721',new),('combined-heldout',old+new)]:
        if rows:report(rows,'commonsenseqa',bank,output/label,label)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['commonsenseqa','gsm8k'],required=True)
    p.add_argument('--original',type=Path,default=Path('outputs/commonsenseqa-angry-mean'))
    p.add_argument('--extension',type=Path,default=Path('outputs/commonsenseqa-remaining721-steering'))
    p.add_argument('--source',type=Path,default=Path('outputs/azure-commonsenseqa500'))
    p.add_argument('--results',nargs='+',type=Path)
    p.add_argument('--conditions',nargs='+',default=['L31-answer-a1','L31-answer-a1-periodic2'])
    p.add_argument('--output',type=Path,default=Path('outputs/independent-scoring-audit-v2'))
    args=p.parse_args()
    if args.task=='commonsenseqa':commonsense_reports(args.original,args.extension,args.source,args.output);return
    if not args.results:p.error('GSM8K needs --results with the saved results.jsonl files to pool')
    cache=args.output/'upstream';RegexFilter=harness_filter(cache)
    source=cache/'gsm8k-test.jsonl'
    if not source.exists():source.write_bytes(download('https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'))
    (cache/'gsm8k-source-sha256.txt').write_text(hashlib.sha256(source.read_bytes()).hexdigest()+'\n')
    bank=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    rows=[];seen=set()
    for path in args.results:
        values=read_rows(path)
        if not values:raise ValueError(f'No saved results: {path}')
        ids={r['question_id'] for r in values}
        if ids&seen:raise ValueError('Input result cohorts overlap; provide one file per cohort')
        seen.update(ids);rows.extend(r for r in values if r['condition'] in ['baseline',*args.conditions])
    report(rows,'gsm8k',bank,args.output/'gsm8k','GSM8K combined',RegexFilter)
    print('GSM8K uses the upstream flexible filter and string exact match, not the harness default prompt/few-shot setup.')
    print('CommonsenseQA choice extraction is an independent audit, not the default text likelihood task.')


if __name__=='__main__':main()
