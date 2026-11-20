"""Combine L0 audio -1 and L30 answer -2 on frozen GSM8K or MultiArith tests."""
import argparse
import fcntl
import json
from pathlib import Path
import statistics

import phi4_all_layer_diagnostic as hooks
from kimi_bbh_layer_diagnostic import digest, freeze, atomic_save
from steer import read_rows

AUDIO = Path('outputs/phi4-sweeps/gsm8k/L0-audio-a-1')
ANSWER = Path('outputs/phi4-gsm-reverse400-869-L30-a-2')
OUTPUT = Path('outputs/phi4-gsm-L0audio-minus1-L30answer-minus2')
CONDITIONS = {
    'neutral': [],
    'audio-steering-only': [dict(layer=0, group='audio', strength=-1.)],
    'answer-steering-only': [dict(layer=30, group='answer', strength=-2.)],
    'combined': [dict(layer=0, group='audio', strength=-1.), dict(layer=30, group='answer', strength=-2.)],
}


def prepare(output, audio=AUDIO, answer=ANSWER, dataset='gsm8k', combined_only=False):
    import torch
    configs = [json.loads((p/'run_config.json').read_text()) for p in (audio, answer)]
    a, b = configs
    for key in ('collection', 'recipe', 'prompt'):
        if a[key] != b[key]: raise ValueError('Standalone sources differ: '+key)
    if a.get('seed',42) != 42 or b.get('seed',42) != 42 or a.get('max_new_tokens',512) != 512 or b.get('max_new_tokens',512) != 512:
        raise ValueError('Expected seed42 and 512-token budget.')
    rows = [r for r in a['recordings'] if r['condition']=='neutral']
    expected={'gsm8k':(400,869),'multiarith':(290,290),'svamp':(500,500)}[dataset]
    if (len(a['collection']),len(rows))!=expected: raise ValueError(f'Expected frozen{expected[0]}/{expected[1]} split.')
    if {r['question_id'] for r in rows} & {r['question_id'] for r in a['collection']}: raise ValueError('Collection/test overlap.')
    old_rows=b['test'] if 'test' in b else [r for r in b['recordings'] if r['condition']=='neutral']
    old = {(r['question_id'],r['voice']):r for r in old_rows}
    for r in rows:
        previous=old[(r['question_id'],r['voice'])]
        if any(r[k]!=previous[k] for k in ('audio_sha256','reference_answer')): raise ValueError('Standalone test differs.')
    vectors=[]; receipts=[]
    for source, config, layer, group in ((audio,a,0,'audio'),(answer,b,30,'answer')):
        state=torch.load(source/'style-vectors.pt',map_location='cpu',weights_only=True)
        origin=Path(config['previous_run']) if config.get('previous_run') else source
        if state['run_config_sha256']!=digest(origin/'run_config.json') or state['collection_question_ids']!=[p['question_id'] for p in a['collection']]:
            raise ValueError('Vector provenance differs.')
        if state['layer']!=layer or state['group']!=group or state['model']!=a['recipe']['model'] or state['revision']!=a['recipe']['revision']:
            raise ValueError('Wrong source vector.')
        vector=state['angry_minus_neutral']
        if vector.shape!=(3072,) or not torch.isfinite(vector).all(): raise ValueError('Invalid vector.')
        vectors.append(vector)
        receipts.append(dict(folder=str(source),config_sha256=digest(source/'run_config.json'),vector_sha256=digest(source/'style-vectors.pt')))
    plan=dict(protocol='phi4-gsm-combined-v1',task='gsm8k',recipe=a['recipe'],prompt=a['prompt'],seed=42,max_new_tokens=512,
        collection=a['collection'],bank=a['bank'],test=rows,conditions=CONDITIONS,sources=receipts,
        note='Exploratory combination after layer/strength sweeps; same disjoint869 questions, no new TTS/collection.')
    if dataset!='gsm8k':
        plan.update(protocol=f'phi4-{dataset}-combined-v1',task=dataset,
            note=f'Exploratory combination after layer/strength sweeps; same disjoint{expected[1]} questions, no new TTS/collection.')
    if combined_only:
        plan['conditions']={k:v for k,v in CONDITIONS.items() if k in ('neutral','combined')}
    output.mkdir(parents=True,exist_ok=True);freeze(output/'run_config.json',plan)
    state=dict(run_config_sha256=digest(output/'run_config.json'),audio=vectors[0],answer=vectors[1])
    path=output/'style-vectors.pt'
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['run_config_sha256']!=state['run_config_sha256'] or any(not torch.equal(saved[k],state[k]) for k in ('audio','answer')):
            raise ValueError('Combined vector changed.')
    else: atomic_save(path,state)
    return plan,state


def completed(plan,output):
    done={}; expected={(r['question_id'],r['voice']):r for r in plan['test']}
    for row in read_rows(output/'results.jsonl'):
        key=row['question_id'],row['voice'],row['condition']; original=expected[key[:2]]
        if key in done or row['run_config_sha256']!=digest(output/'run_config.json') or row['interventions']!=plan['conditions'][key[2]]:
            raise ValueError('Duplicate or incompatible saved result.')
        for field in ('audio_sha256','reference_answer'):
            if row[field]!=original[field]: raise ValueError('Saved input differs.')
        if (row['prompt']!=plan['prompt'] or row['seed']!=42 or row['max_new_tokens']!=512 or
                row['model']!=plan['recipe']['model'] or row['revision']!=plan['recipe']['revision']):
            raise ValueError('Saved generation recipe differs.')
        done[key]=row
    return done


def compatible(row, source_hash, plan, original, interventions):
    return (row['run_config_sha256']==source_hash and row['prompt']==plan['prompt'] and
        row['seed']==42 and row['max_new_tokens']==512 and row['model']==plan['recipe']['model'] and
        row['revision']==plan['recipe']['revision'] and row['interventions']==interventions and
        all(row[k]==original[k] for k in ('audio_sha256','reference_answer')))


def run(plan,output,vectors):
    done=completed(plan,output); originals={(r['question_id'],r['voice']):r for r in plan['test']}
    config_hash=digest(output/'run_config.json')
    with (output/'results.jsonl').open('a') as stream:
        for source in plan['sources']:
            folder=Path(source['folder'])
            for row in read_rows(folder/'results.jsonl'):
                pair=row['question_id'],row['voice']
                if pair not in originals: continue
                for label in (c for c in plan['conditions'] if c!='combined'):
                    key=(*pair,label)
                    if key in done or not compatible(row,source['config_sha256'],plan,originals[pair],plan['conditions'][label]): continue
                    if row.get('strength',0) and row.get('vector_sha256')!=source['vector_sha256']: raise ValueError('Control vector differs.')
                    saved=dict(row,condition=label,run_config_sha256=config_hash,reused_from=str(folder))
                    stream.write(json.dumps(saved)+'\n');stream.flush();done[key]=saved
        pending=[(r,c) for r in plan['test'] for c in plan['conditions'] if (r['question_id'],r['voice'],c) not in done]
        print(f'Reused/resumed {len(done)} answers; {len(pending)} generations pending. No GPU queue wait.',flush=True)
        if not pending:return
        if len(plan['conditions'])==2 and any(label!='combined' for _,label in pending):
            raise ValueError('Combined-only run requires existing compatible neutral answers; no baseline generations started.')
        from evaluate_bbh_candidates import concurrent_budget_now
        concurrent_budget_now(18.)
        native=hooks.engine.NativePhi()
        (output/'runtime.json').write_text(json.dumps(dict(hooks_sha256=digest(hooks.__file__),
            runner_sha256=digest(__file__),policy='No GPU queue;18 GiB Torch cap'),indent=2)+'\n')
        for i,(row,label) in enumerate(pending,1):
            changes=plan['conditions'][label]
            intervention=[dict(x,direction=vectors[x['group']]) for x in changes]
            print(f'[{i}/{len(pending)}] {row["question_id"]} {label}',flush=True)
            trace=hooks.generate(native,row,plan['prompt'],interventions=intervention,max_new_tokens=512)
            saved=dict(row,**trace,condition=label,interventions=changes,prompt=plan['prompt'],seed=42,max_new_tokens=512,
                model=plan['recipe']['model'],revision=plan['recipe']['revision'],run_config_sha256=config_hash,
                vector_sha256=digest(output/'style-vectors.pt'))
            stream.write(json.dumps(saved)+'\n');stream.flush();print(trace['response'],flush=True)


def report(plan,output):
    from rescore_audio_steering import HarnessTaskScorer,numeric_equal,paired
    rows=list(completed(plan,output).values())
    if not rows: print('No answers yet.');return
    scorer=HarnessTaskScorer('gsm8k',output/'harness/upstream')
    scores=scorer.score([r['response'] for r in rows],[dict(question=plan['bank'][r['question_id']]['question'],answer=r['reference_answer']) for r in rows])
    for i,row in enumerate(rows):
        row['scores']={m:v[i]['correct'] for m,v in scores.items()}
        row['scores']['format-compliant']=scores['strict-match'][i]['prediction']!='[invalid]'
        row['scores']['flexible-numeric']=numeric_equal(scores['flexible-extract'][i]['prediction'],row['reference_answer'])
    groups={c:{(r['question_id'],r['voice']):r for r in rows if r['condition']==c} for c in plan['conditions']}
    keys=sorted(set.intersection(*(set(g) for g in groups.values())))
    print(f'{len(keys)}/{len(plan["test"])} questions complete in all {len(groups)} conditions; exploratory, unadjusted paired tests.')
    comparisons=[]
    if keys:
        for c,g in groups.items():
            tokens=[g[k]['response_tokens'] for k in keys]
            print(c,'mean/median tokens:',round(statistics.mean(tokens),1),statistics.median(tokens))
        for metric in rows[0]['scores']:
            for c,g in groups.items():print(f'{metric} / {c}: {sum(g[k]["scores"][metric] for k in keys)}/{len(keys)}')
            for ref in (c for c in groups if c!='combined'):
                result=paired([(groups[ref][k]['scores'][metric],groups['combined'][k]['scores'][metric]) for k in keys])
                comparisons.append(dict(metric=metric,reference=ref,condition='combined',**result))
                print(f'combined vs {ref} / {metric}: gain={result["gain_pp"]:+.2f}; p={result["mcnemar_exact_p"]:.4g}; CI={result["ci95_gain_pp"]}')
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (output/'summary.json').write_text(json.dumps(dict(n=len(keys),paired=comparisons,scorer=scorer.provenance),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('run','summary'),default='run');p.add_argument('--output',type=Path);p.add_argument('--dataset',choices=('gsm8k','multiarith','svamp'),default='gsm8k')
    args=p.parse_args()
    if args.output is None:
        args.output=OUTPUT if args.dataset=='gsm8k' else Path(f'outputs/phi4-{args.dataset}-L0audio-minus1-L30answer-minus2'+('-combined-only' if args.dataset=='svamp' else ''))
    if args.mode=='summary':report(json.loads((args.output/'run_config.json').read_text()),args.output);return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.dataset=='gsm8k':plan,vectors=prepare(args.output)
        elif args.dataset=='svamp':plan,vectors=prepare(args.output,Path('outputs/phi4-sweeps/svamp/L0-audio-a-1'),
            Path('outputs/phi4-sweeps/svamp/L30-answer-a-1'),dataset='svamp',combined_only=True)
        else:plan,vectors=prepare(args.output,Path('outputs/phi4-sweeps/multiarith/L0-audio-a-1'),
            Path('outputs/phi4-multiarith-reverse290-290-L30'),dataset='multiarith')
        run(plan,args.output,vectors)

if __name__=='__main__':main()
