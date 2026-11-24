"""Object Counting: neutral and angry with the approved 'Supply reasoning' prompt."""
import argparse
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
from statistics import mean, median
import subprocess
import time

import run_object_counting_prompts as base
from rescore_audio_steering import HarnessTaskScorer, paired

PROTOCOL = 'diffa-object-counting-supply-reasoning-v1'
PROMPT = ('Answer the question in the audio. Supply reasoning before the final answer. '
          'End your response with a final answer line in the format:\nThe answer is <number>.')
CONDITIONS = {'neutral-supply-reasoning':'neutral', 'angry-supply-reasoning':'angry'}
winners = base.winners


def parent_plan(parent, child=None):
    config=parent/'run_config.json'
    old=json.loads(config.read_text())
    ids=old['question_order']
    if (old['protocol']!=base.PROTOCOL or old['prompts']!=base.PROMPTS
            or old['recipe']!=base.evaluation.night.vanilla.RECIPE
            or old['source_git_blob']!=base.SOURCE_BLOB or len(ids)!=250 or len(set(ids))!=250
            or set(ids)!=set(old['bank']) or len(old['records'])!=1250
            or {(r['question_id'],r['condition']) for r in old['records']}!={(q,c) for q in ids for c in base.CONDITIONS}):
        raise ValueError('Need the unchanged original five-condition Object Counting configuration.')
    if child is not None and winners.digest(config)!=child['parent_config_sha256']:
        raise ValueError('Parent configuration changed after preparation.')
    return old


def prepare(parent,output):
    old=parent_plan(parent)
    records={(r['question_id'],r['condition']):r for r in old['records']}
    selected=[]
    for q in old['question_order']:
        for condition,style in CONDITIONS.items():
            r=records[q,style]
            if r['style']!=style or r['voice']!=base.VOICE or r['reference_answer']!=old['bank'][q]['answer']:
                raise ValueError('Source recording voice/style/reference differs.')
            selected.append(dict(r,condition=condition))
    plan=dict(protocol=PROTOCOL,name=base.TASK,task=base.TASK,model='diffa',bank=old['bank'],
        question_order=old['question_order'],seed=42,max_new_tokens=512,recipe=old['recipe'],
        prompt=PROMPT,prompts={c:PROMPT for c in CONDITIONS},conditions=list(CONDITIONS),records=selected,
        parent_config_sha256=winners.digest(parent/'run_config.json'),
        adaptation='Two additional conditions: identical existing neutral and angry recordings with the approved '
                   'Supply reasoning prompt. Decoder and final-answer format unchanged. No synthesis, examples, '
                   'classification, fitting or steering; previous configurations/responses remain separate.')
    winners.freeze(output/'run_config.json',plan)
    winners.atomic_text(output/'prompt.txt',PROMPT+'\n')
    print('Object Counting: neutral + angry, Supply reasoning prompt; 250 questions, 500 new generations.',flush=True)
    print('Prompt:\n'+PROMPT,flush=True)
    return plan


def report(output,parent,old_angry):
    plan=json.loads((output/'run_config.json').read_text())
    old=parent_plan(parent,plan)
    previous=base.raw_rows(parent,old)
    current=base.raw_rows(output,plan)
    rows=[r for r in previous.values() if r['condition'] in ('neutral','neutral-reasoning','angry')]+list(current.values())
    extra_config=old_angry/'run_config.json'
    if extra_config.exists():
        extra=json.loads(extra_config.read_text())
        if extra.get('parent_config_sha256')!=plan['parent_config_sha256']:
            raise ValueError('Previous angry-reasoning condition belongs to a different parent.')
        rows+=list(base.raw_rows(old_angry,extra).values())
    scorer=HarnessTaskScorer('bbh_cot_zeroshot_object_counting',output/'scoring')
    docs=[dict(input=plan['bank'][r['question_id']]['original_input'],target=plan['bank'][r['question_id']]['original_target']) for r in rows]
    values=scorer.score([r['response'] for r in rows],docs)
    scored=[dict(r,metrics={m:v[i]['correct'] for m,v in values.items()},
                 predictions={m:v[i]['prediction'] for m,v in values.items()}) for i,r in enumerate(rows)]
    order=('neutral','neutral-reasoning','angry','angry-reasoning',*CONDITIONS)
    groups={c:{r['question_id']:r for r in scored if r['condition']==c} for c in order}
    lengths,accuracy,comparisons={},{},[]
    print(f'\nObject Counting Supply reasoning: {len(current)}/500 completed.',flush=True)
    for c,g in groups.items():
        if not g:continue
        tokens=[r['response_tokens'] for r in g.values()]
        lengths[c]=dict(n=len(g),mean_tokens=mean(tokens),median_tokens=median(tokens))
        accuracy[c]={m:dict(n=len(g),correct=sum(r['metrics'][m] for r in g.values())) for m in values}
        print(f'{c}: N={len(g)}; mean/median tokens={mean(tokens):.1f}/{median(tokens):g}',flush=True)
        for m,a in accuracy[c].items():print(f'  {m}: {a["correct"]}/{a["n"]} ({a["correct"]/a["n"]:.1%})',flush=True)
    contrasts=(('neutral-supply-reasoning','angry-supply-reasoning'),
               ('neutral','neutral-supply-reasoning'),('angry','angry-supply-reasoning'),
               ('neutral-reasoning','neutral-supply-reasoning'),('angry-reasoning','angry-supply-reasoning'))
    for reference,test in contrasts:
        ids=sorted(groups[reference].keys()&groups[test].keys())
        if not ids:continue
        lengths_matched={c:dict(mean_tokens=mean(groups[c][q]['response_tokens'] for q in ids),
                               median_tokens=median(groups[c][q]['response_tokens'] for q in ids)) for c in (reference,test)}
        for metric in values:
            stats=paired([(groups[reference][q]['metrics'][metric],groups[test][q]['metrics'][metric]) for q in ids])
            comparisons.append(dict(reference=reference,test=test,metric=metric,matched_lengths=lengths_matched,**stats))
            print(f'{test} vs {reference} / {metric}: SAME N={len(ids)}; gain={stats["gain_pp"]:+.2f} pp; '
                  f'p={stats["mcnemar_exact_p"]:.4g}; CI={stats["ci95_gain_pp"]}',flush=True)
    winners.atomic_text(output/'delivery-rescored.jsonl',''.join(json.dumps(r)+'\n' for r in scored))
    winners.atomic_text(output/'delivery-summary.json',json.dumps(dict(protocol=PROTOCOL,accuracy=accuracy,
        lengths=lengths,comparisons=comparisons,scorer=scorer.provenance,
        statistics='Matched question IDs; exact McNemar p and paired bootstrap95% CI, unadjusted.'),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('run','prepare','summary'),default='run')
    parser.add_argument('--parent',type=Path,default=Path('outputs/diffa-object-counting-prompts'))
    parser.add_argument('--old-angry',type=Path,default=Path('outputs/diffa-object-counting-angry-reasoning'))
    parser.add_argument('--output',type=Path,default=Path('outputs/diffa-object-counting-supply-reasoning'))
    args=parser.parse_args()
    if args.mode=='summary':
        if not (args.output/'run_config.json').exists():print('Supply reasoning run: not prepared yet.',flush=True)
        else:report(args.output,args.parent,args.old_angry)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'runner.lock').open('a') as own:
        try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Supply reasoning runner already active.')
        plan=prepare(args.parent,args.output)
        base.scoring_preflight(args.output)
        if args.mode=='prepare':return
        done=base.raw_rows(args.output,plan)
        if len(done)<len(plan['records']):
            for r in plan['records']:
                if winners.digest(r['audio_file'])!=r['audio_sha256']:
                    raise ValueError('Existing neutral/angry recording missing or changed; restore its original bytes.')
            with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
                print('Waiting for shared GPU lock and idle GPU; no TTS or Azure credentials needed.',flush=True)
                fcntl.flock(gpu,fcntl.LOCK_EX)
                while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(30)
                backend=base.evaluation.night.DiffaBackend()
                from src.dataloader import build_chat_prompt
                from diffa_model_receipts import model_revisions
                text,count=base.evaluation.context_guard(backend.tokenizer,PROMPT,build_chat_prompt)
                winners.atomic_text(args.output/'native-chat-prompt.txt',text)
                revisions=model_revisions();packages={k:version(k) for k in ('torch','transformers')}
                original_runtime=args.parent/'runtime.json'
                if original_runtime.exists():
                    old=json.loads(original_runtime.read_text())
                    if old['model_revisions']!=revisions or old['packages']!=packages:
                        raise ValueError('Model revisions/package versions differ from the original conditions.')
                winners.freeze(args.output/'runtime.json',dict(model_revisions=revisions,packages=packages,
                    model_config=backend.model.config.to_dict(),native_prompt_tokens=count,runner_sha256=winners.digest(__file__)))
                base.generate(plan,args.output,backend)
        report(args.output,args.parent,args.old_angry)


if __name__=='__main__':main()
