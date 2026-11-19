"""Phi IFEval: preserve the original508-question spoken cohort and official checks."""
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

from kimi_bbh_layer_diagnostic import digest,freeze
from run_kimi_gsm8k import prepare_audio
import run_reasoning_transfer as audio
import run_spoken_ifeval as official
from steer import read_rows

SPLIT=Path(__file__).with_name('configs')/'ifeval-spoken508-split.json'


def prepare(root):
    split=json.loads(SPLIT.read_text())
    if split['official_revision']!=official.REVISION:raise ValueError('Official IFEval revision differs.')
    train,test=split['collection_ids'],split['test_ids']
    if len(train)!=254 or len(test)!=254 or len(set(train+test))!=508:raise ValueError('Expected disjoint254/254 cohort.')
    package=official.setup_official();source_file=package/'data/input_data.jsonl'
    examples={f'ifeval_{e["key"]}':e for e in read_rows(source_file)}
    questions=[dict(question_id=q,question=examples[q]['prompt'],answer=None,**examples[q]) for q in train+test]
    source=root/'ifeval/audio';source.mkdir(parents=True,exist_ok=True);audio.bank_file(source/'questions.jsonl',questions)
    freeze(root/'ifeval/selection.json',dict(split_sha256=digest(SPLIT),**split))
    prepare_audio(SimpleNamespace(source=source,audio_sources=[Path('outputs/azure-spoken-ifeval'),Path('outputs/spoken-ifeval-diffa2')]),
                  dict(collection=[],test=questions))
    bank={q['question_id']:q for q in questions};records=audio.source_records(source,bank,['neutral','angry']);pairs={}
    for q in train+test:
        pair=dict(question_id=q)
        for style in ('neutral','angry'):
            row=dict(records[q,style],reference_answer=None);row.pop('answer');pair[style]=row
        if pair['neutral']['voice']!=pair['angry']['voice']:raise ValueError('IFEval voices differ.')
        pair['voice']=pair['neutral']['voice'];pairs[q]=pair
    data=dict(protocol='phi4-ifeval-data-v1',task='ifeval',prompt=official.PROMPT,seed=42,max_new_tokens=1024,
        collection=[pairs[q] for q in train],test=[pairs[q] for q in test],bank=bank,
        official_revision=official.REVISION,official_source_sha256=digest(source_file),frozen_split_sha256=digest(SPLIT),
        selection='Original508 spoken cohort; same254/254 split; no new duration/outcome/classifier filtering.')
    freeze(root/'ifeval/data_config.json',data)
    print('IFEval:254 collection /254 test; original spoken split;1024-token native Phi budget.',flush=True)
    return data


def report(plan,output,rows):
    official.setup_official();evaluator=official.official_scorer();scored=[]
    for row in rows:scored.append(dict(row,official_scores=official.score(plan['bank'][row['question_id']],row['response'],evaluator)))
    labels=list(dict.fromkeys(r['condition'] for r in plan['recordings']))
    groups={c:{(r['question_id'],r['voice']):r for r in scored if r['condition']==c} for c in labels}
    keys=sorted(set.intersection(*(set(g) for g in groups.values())))
    print(f'Official IFEval checks: {len(keys)}/{len(plan["bank"])} questions completed in all {len(groups)} conditions; spoken adaptation.')
    print('Condition                            N Strict prompt Loose prompt Strict instruction Loose instruction Mean tokens Median tokens')
    totals=[];comparisons=[]
    for label,g in groups.items():
        if not keys:continue
        r=dict(condition=label,n=len(keys))
        for mode in ('strict','loose'):
            bits=[b for k in keys for b in g[k]['official_scores'][mode]['instructions']]
            r[mode+'_prompt']=sum(g[k]['official_scores'][mode]['prompt'] for k in keys)/len(keys)
            r[mode+'_instruction']=sum(bits)/len(bits)
        lengths=[g[k]['response_tokens'] for k in keys];r.update(mean_tokens=statistics.mean(lengths),median_tokens=statistics.median(lengths));totals.append(r)
        print(f'{label:36} {len(keys):3d} '+ ' '.join(f'{100*r[m]:12.1f}%' for m in ('strict_prompt','loose_prompt','strict_instruction','loose_instruction'))+f' {r["mean_tokens"]:11.1f} {r["median_tokens"]:13.1f}')
    for mode in ('strict','loose'):
        for label in labels:
            if label=='neutral' or not keys:continue
            result=official.paired([(groups['neutral'][k]['official_scores'][mode]['prompt'],groups[label][k]['official_scores'][mode]['prompt']) for k in keys])
            comparisons.append(dict(metric=mode+'_prompt',condition=label,**result))
            print(f'{label} /{mode}_prompt: gain={result["gain_pp"]:+.2f} pp; W/L={result["wins"]}/{result["losses"]}; p={result["mcnemar_exact_p"]:.4g}; CI={result["ci95_gain_pp"]}')
    (output/'official-scored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    (output/'summary.json').write_text(json.dumps(dict(primary_metric='strict_prompt',official_revision=official.REVISION,totals=totals,paired=comparisons,
        note='Official prompt/instruction compliance; no factual-correctness judgment. Exploratory/unadjusted.'),indent=2)+'\n')
