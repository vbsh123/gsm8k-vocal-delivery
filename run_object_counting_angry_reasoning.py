"""Add angry + the existing neutral reasoning prompt to Object Counting; reuse all audio."""
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

PROTOCOL = 'diffa-object-counting-angry-reasoning-v1'
CONDITION = 'angry-reasoning'
winners = base.winners


def prepare(parent, output):
    config = parent/'run_config.json'
    old = json.loads(config.read_text())
    ids = old['question_order']
    if (old['protocol'] != base.PROTOCOL or old['prompts'] != base.PROMPTS
            or old['recipe'] != base.evaluation.night.vanilla.RECIPE
            or old['source_git_blob'] != base.SOURCE_BLOB or len(ids) != 250
            or len(set(ids)) != 250 or set(ids) != set(old['bank'])
            or len(old['records']) != 1250
            or {(r['question_id'], r['condition']) for r in old['records']} != {(q,c) for q in ids for c in base.CONDITIONS}):
        raise ValueError('Need the unchanged five-condition Object Counting source configuration.')
    angry = {r['question_id']: r for r in old['records'] if r['condition'] == 'angry'}
    if any(r['style'] != 'angry' or r['voice'] != base.VOICE
           or r['reference_answer'] != old['bank'][q]['answer'] for q,r in angry.items()):
        raise ValueError('Parent angry audio identity differs.')
    plan = dict(protocol=PROTOCOL, name=base.TASK, model='diffa', task=base.TASK,
        bank=old['bank'], question_order=ids, seed=42, max_new_tokens=512, recipe=old['recipe'],
        prompt=old['prompts']['neutral-reasoning'], prompts={CONDITION:old['prompts']['neutral-reasoning']},
        parent_config_sha256=winners.digest(config), parent_protocol=old['protocol'],
        records=[dict(angry[q], condition=CONDITION) for q in ids],
        adaptation='One additional condition: existing angry recordings, identical prompt and decoder to '
                   'neutral-reasoning. No synthesis, examples, classification, fitting or steering.')
    winners.freeze(output/'run_config.json',plan)
    winners.atomic_text(output/'prompt.txt',plan['prompt']+'\n')
    print('Object Counting: 250 angry + reasoning generations; reusing existing angry audio.',flush=True)
    print('Prompt:\n'+plan['prompt'],flush=True)
    return plan


def parent_plan(parent, plan):
    config=parent/'run_config.json'
    if winners.digest(config)!=plan['parent_config_sha256']:
        raise ValueError('Parent configuration changed after this condition was prepared.')
    return json.loads(config.read_text())


def report(output,parent):
    plan=json.loads((output/'run_config.json').read_text())
    old=parent_plan(parent,plan)
    # Read parent answers only: no rescoring files/configs in its directory are modified.
    previous=base.raw_rows(parent,old)
    current=base.raw_rows(output,plan)
    rows=[r for r in previous.values() if r['condition'] in ('neutral','neutral-reasoning','angry')]
    rows+=list(current.values())
    scorer=HarnessTaskScorer('bbh_cot_zeroshot_object_counting',output/'scoring')
    docs=[dict(input=plan['bank'][r['question_id']]['original_input'],
               target=plan['bank'][r['question_id']]['original_target']) for r in rows]
    scores=scorer.score([r['response'] for r in rows],docs)
    scored=[dict(r,metrics={m:v[i]['correct'] for m,v in scores.items()},
                 predictions={m:v[i]['prediction'] for m,v in scores.items()}) for i,r in enumerate(rows)]
    conditions=('neutral','neutral-reasoning','angry',CONDITION)
    groups={c:{r['question_id']:r for r in scored if r['condition']==c} for c in conditions}
    lengths,accuracy,comparisons={},{},[]
    print(f'\nObject Counting angry + reasoning: {len(current)}/250 completed.',flush=True)
    for condition,group in groups.items():
        if not group:continue
        tokens=[r['response_tokens'] for r in group.values()]
        lengths[condition]=dict(n=len(tokens),mean_tokens=mean(tokens),median_tokens=median(tokens))
        accuracy[condition]={m:dict(n=len(group),correct=sum(r['metrics'][m] for r in group.values())) for m in scores}
        print(f'{condition}: N={len(group)}; mean/median tokens={mean(tokens):.1f}/{median(tokens):g}',flush=True)
        for m,value in accuracy[condition].items():
            print(f'  {m}: {value["correct"]}/{value["n"]} ({value["correct"]/value["n"]:.1%})',flush=True)
    for reference in ('neutral-reasoning','angry','neutral'):
        ids=sorted(groups[reference].keys()&groups[CONDITION].keys())
        if not ids:continue
        matched={c:dict(mean_tokens=mean(groups[c][q]['response_tokens'] for q in ids),
                       median_tokens=median(groups[c][q]['response_tokens'] for q in ids)) for c in (reference,CONDITION)}
        for metric in scores:
            stats=paired([(groups[reference][q]['metrics'][metric],groups[CONDITION][q]['metrics'][metric]) for q in ids])
            comparisons.append(dict(reference=reference,test=CONDITION,metric=metric,matched_lengths=matched,**stats))
            print(f'{CONDITION} vs {reference} / {metric}: SAME N={len(ids)}; gain={stats["gain_pp"]:+.2f} pp; '
                  f'p={stats["mcnemar_exact_p"]:.4g}; CI={stats["ci95_gain_pp"]}',flush=True)
    winners.atomic_text(output/'delivery-rescored.jsonl',''.join(json.dumps(r)+'\n' for r in scored))
    winners.atomic_text(output/'delivery-summary.json',json.dumps(dict(protocol=PROTOCOL,accuracy=accuracy,
        lengths=lengths,comparisons=comparisons,scorer=scorer.provenance,
        statistics='Matched question IDs; exact McNemar p and paired bootstrap95% CI, unadjusted.'),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('run','prepare','summary'),default='run')
    parser.add_argument('--parent',type=Path,default=Path('outputs/diffa-object-counting-prompts'))
    parser.add_argument('--output',type=Path,default=Path('outputs/diffa-object-counting-angry-reasoning'))
    args=parser.parse_args()
    if args.mode=='summary':
        if not (args.output/'run_config.json').exists():
            print('Angry + reasoning: not prepared yet.',flush=True)
        else:report(args.output,args.parent)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'runner.lock').open('a') as own:
        try:fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Angry + reasoning runner already active.')
        plan=prepare(args.parent,args.output)
        base.scoring_preflight(args.output)
        if args.mode=='prepare':return
        done=base.raw_rows(args.output,plan)
        if len(done)<250:
            for r in plan['records']:
                if winners.digest(r['audio_file'])!=r['audio_sha256']:
                    raise ValueError('Existing angry recording missing/changed. Restore its original bytes; do not regenerate.')
            with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
                print('Waiting for shared GPU lock and idle GPU; no TTS or Azure credentials needed.',flush=True)
                fcntl.flock(gpu,fcntl.LOCK_EX)
                while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                    time.sleep(30)
                backend=base.evaluation.night.DiffaBackend()
                from src.dataloader import build_chat_prompt
                from diffa_model_receipts import model_revisions
                text,count=base.evaluation.context_guard(backend.tokenizer,plan['prompt'],build_chat_prompt)
                winners.atomic_text(args.output/'native-chat-prompt.txt',text)
                revisions=model_revisions()
                packages={k:version(k) for k in ('torch','transformers')}
                original_runtime=args.parent/'runtime.json'
                if original_runtime.exists():
                    old=json.loads(original_runtime.read_text())
                    if old['model_revisions']!=revisions or old['packages']!=packages:
                        raise ValueError('Model revisions/package versions differ from the original five conditions.')
                winners.freeze(args.output/'runtime.json',dict(model_revisions=revisions,packages=packages,
                    model_config=backend.model.config.to_dict(),native_prompt_tokens=count,
                    runner_sha256=winners.digest(__file__)))
                base.generate(plan,args.output,backend)
        report(args.output,args.parent)


if __name__=='__main__':main()
