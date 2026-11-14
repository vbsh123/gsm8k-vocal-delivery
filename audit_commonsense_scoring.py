"""Compare strict/relaxed CommonsenseQA scores and export two opposite neutral/angry examples."""

import argparse
from collections import defaultdict
import csv
from html import escape
import json
from pathlib import Path
import re
import zipfile

from evaluate_commonsense_audio import score_choice,STYLES
from steer import read_rows


def normalized(text):
    return re.sub(r'\W+',' ',text.casefold()).strip()


def relaxed_choice(response,choices):
    """Extract without receiving the gold answer; never search reasoning for it."""
    clean=response.replace('**','').replace('__','').replace('`','')
    clean=re.sub(r'\\boxed\{([^{}]*)\}',r'\1',clean)
    labels=list(re.finditer(r'(?:final\s+(?:answer|choice)|(?:the\s+)?(?:correct\s+)?(?:answer|choice))\s*(?::|is\b)',clean,re.I))
    lines=(clean[labels[-1].end():].strip().splitlines() if labels else [l.strip() for l in clean.splitlines() if l.strip()][-1:])
    if not lines:return None
    answer=lines[0].strip().strip('$')
    match=re.match(r'^(?:(?:option|choice)\s+)?[\(\[]?([A-E])[\)\]]?(?=$|[\s.,:;!/?–—-])',answer,re.I)
    by_text=defaultdict(list)
    for choice in choices:by_text[normalized(choice['text'])].append(choice['label'])
    if match:
        letter=match[1].upper();suffix=answer[match.end():]
        if re.match(r'\s*(?:or|and|/|,)\s*(?:(?:option|choice)\s+)?[\(\[]?[A-E]\b',suffix,re.I):return None
        remainder=normalized(suffix)
        if not remainder:return letter
        if re.match(r'^[\s.,;:/–—-]+(?:(?:option|choice)\s+)?[A-E](?=$|[\s.):])',suffix,re.I):return None
        mapped=by_text.get(remainder,[])
        if mapped:return letter if mapped==[letter] else None
        if re.fullmatch(r'[A-E](?:\s+(?:or|and)\s+[A-E])*',remainder,re.I):return None
        # With an explicit answer label, explanatory prose after one letter is allowed.
        return letter if labels else None
    mapped=by_text.get(normalized(answer),[])
    return mapped[0] if len(mapped)==1 else None


def scored(row,bank):
    question=bank[row['question_id']]
    strict=score_choice(row['response'],row['reference_answer'])
    relaxed=relaxed_choice(row['response'],question['choices'])
    return dict(strict=strict['correct'],relaxed=relaxed==row['reference_answer'],
                strict_parse_failed=strict['parse_failed'],relaxed_parse_failed=relaxed is None,
                strict_prediction=strict['predicted_answer'],relaxed_prediction=relaxed)


def accuracy_table(title,groups,bank):
    print('\n'+title)
    print(f'{"Condition":25} {"N":>5} {"Strict":>9} {"Relaxed":>9} {"Δ pp":>7} {"Strict parse fail":>17} {"Relaxed parse fail":>18}')
    summary=[]
    for label,rows in groups:
        if not rows:continue
        values=[scored(r,bank) for r in rows];n=len(values)
        a=sum(v['strict'] for v in values);b=sum(v['relaxed'] for v in values)
        if any(v['strict'] and not v['relaxed'] for v in values):raise ValueError('Relaxed scorer rejected a strict-correct answer; inspect scorer')
        sf=sum(v['strict_parse_failed'] for v in values);rf=sum(v['relaxed_parse_failed'] for v in values)
        print(f'{label:25} {n:5} {a/n:9.1%} {b/n:9.1%} {100*(b-a)/n:+7.2f} {sf:17} {rf:18}')
        summary.append(dict(table=title,condition=label,n=n,strict_correct=a,relaxed_correct=b,
                            strict_parse_failed=sf,relaxed_parse_failed=rf))
    return summary


def examples(rows,bank,source,output):
    groups=defaultdict(dict)
    for row in rows:
        if row['style'] in ('neutral','angry'):groups[row['question_id'],row['voice']][row['style']]=row
    pages=[];found=[]
    archive=output/'neutral-angry-examples.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for title,neutral_correct,angry_correct in [('Neutral wrong; angry correct',False,True),('Neutral correct; angry wrong',True,False)]:
            selected=None
            for key,clips in sorted(groups.items()):
                if set(clips)!=set(('neutral','angry')):continue
                n=scored(clips['neutral'],bank);a=scored(clips['angry'],bank)
                if any(v['strict_parse_failed'] or v['relaxed_parse_failed'] for v in (n,a)):continue
                if n['strict']==n['relaxed']==neutral_correct and a['strict']==a['relaxed']==angry_correct:
                    selected=clips;break
            if selected is None:
                print(title+': no example with agreement between strict and relaxed scoring.');continue
            question=bank[selected['neutral']['question_id']]
            print('\n'+title+' (strict and relaxed agree)')
            print(question['question_id']);print(question.get('question_stem',question['question']))
            print('Choices:',question['choices']);print('Gold:',selected['neutral']['reference_answer'])
            parts=[f'<h2>{escape(title)}</h2><p>{escape(question["question_id"])}</p>',
                   f'<pre>{escape(question["question"])}</pre><p>Gold: {escape(selected["neutral"]["reference_answer"])}</p>']
            record=dict(category=title,question_id=question['question_id'],clips={})
            for style in ('neutral','angry'):
                row=selected[style];path=source/row['audio_path']
                print(style,':',path);print(row['response'])
                z.write(path,arcname=path.name)
                parts.append(f'<h3>{style}</h3><audio controls src="{escape(path.name,quote=True)}"></audio><pre>{escape(row["response"])}</pre>')
                record['clips'][style]=dict(audio_path=str(path),response=row['response'],**scored(row,bank))
            found.append(record);pages.extend(parts)
        z.writestr('listen.html','<!doctype html><meta charset="utf-8"><title>Neutral versus angry examples</title>'
                   '<style>body{font:16px system-ui;max-width:950px;margin:30px auto}pre{white-space:pre-wrap}audio{width:100%}</style>'+''.join(pages))
    (output/'examples.json').write_text(json.dumps(found,indent=2)+'\n')
    print('\nListening examples:',archive)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/azure-commonsenseqa500'))
    p.add_argument('--diffa',type=Path,default=Path('outputs/commonsenseqa-diffa2'))
    p.add_argument('--steering',type=Path,default=Path('outputs/commonsenseqa-angry-mean'))
    p.add_argument('--output',type=Path,default=Path('outputs/commonsenseqa-scoring-audit'))
    p.add_argument('--export-examples',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    bank={r['question_id']:r for r in read_rows(args.source/'questions.jsonl')}
    if not bank:raise ValueError('Question/choice metadata is missing')
    summary=[];diffa=read_rows(args.diffa/'results.jsonl')
    if not diffa:raise ValueError('No saved DIFFA results')
    for model,rows in [('DIFFA',diffa),('Qwen',read_rows(args.source/'results-qwen-commonsenseqa.jsonl')),
                       ('Kimi',read_rows(args.source/'results-kimi-commonsenseqa.jsonl'))]:
        latest={r['audio_path']:r for r in rows}
        if not latest:
            print(model+': no saved CommonsenseQA answers here.');continue
        summary+=accuracy_table(model+' — all completed unsteered answers',
            [(s,[r for r in latest.values() if r['style']==s]) for s in STYLES],bank)
    steering=args.steering
    if not (steering/'results.jsonl').exists() and steering.name=='commonsenseqa-angry-mean':
        steering=steering.with_name('commonsenseqa-L31-mean')
    saved={(r['clip_id'],r['condition']):r for r in read_rows(steering/'results.jsonl')}
    keys={(r['question_id'],r['voice']) for r in saved.values() if r['condition']=='baseline'}
    if keys:
        latest={r['audio_path']:r for r in diffa}
        groups=[(s,[r for r in latest.values() if r['style']==s and (r['question_id'],r['voice']) in keys]) for s in STYLES]
        groups+=[(c,[r for r in saved.values() if r['condition']==c]) for c in sorted({r['condition'] for r in saved.values()}-{'baseline'})]
        summary+=accuracy_table('DIFFA — steering test questions only; compare these rows together',groups,bank)
    else:print('No steering baselines found at',steering)
    if summary:
        with (args.output/'accuracy.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    if args.export_examples:examples(diffa,bank,args.source,args.output)
    print('Relaxed scoring accepts explicit letters with prose or exact option text. Ambiguous answers remain incorrect.')
    print('No model was loaded, no generations were run, and original answers/scores were not changed.')


if __name__=='__main__':main()
