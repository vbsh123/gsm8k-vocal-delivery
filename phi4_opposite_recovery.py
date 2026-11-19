"""Opposite GSM recovery diagnostic: neutral-correct/angry-wrong; negative steering on both inputs."""
import argparse
import json
from pathlib import Path

import phi4_all_layer_diagnostic as diagnostic
from kimi_bbh_layer_diagnostic import digest, freeze
from steer import read_rows

DEFAULT_SOURCE=Path('outputs/phi4-gsm400-all-layers-test869-minus1')
DEFAULT_OUTPUT=Path('outputs/phi4-gsm400-opposite-all-layers-minus1')


def select_cases(rows):
    groups={}
    for r in rows:groups.setdefault((r['question_id'],r['voice']),{})[r['style']]=r
    return [dict(question_id=q,voice=v) for (q,v),g in sorted(groups.items())
        if set(g)=={'neutral','angry'} and g['neutral']['scores']['flexible-numeric']
        and not g['angry']['scores']['flexible-numeric']]


def prepare(source,output):
    plan=json.loads((source/'run_config.json').read_text());tasks,done=diagnostic.references(plan,source)
    if len(tasks)!=len(done):raise ValueError('Finish original neutral/angry references first.')
    rows,_=diagnostic.score_rows(plan,list(done.values()),source)
    cases=select_cases(rows);train={r['question_id'] for r in plan['collection']}
    if train & {r['question_id'] for r in cases}:raise ValueError('Collection/test overlap.')
    output.mkdir(parents=True,exist_ok=True)
    frozen=dict(protocol='phi4-opposite-recovery-all32-minus1-v1',evaluation_kind='opposite-recovery-diagnostic',
        source=str(source.absolute()),source_config_sha256=digest(source/'run_config.json'),
        references_sha256=diagnostic.reference_digest(source),vector=str((source/'style-vectors.pt').absolute()),
        vector_sha256=digest(source/'style-vectors.pt'),cases=cases,layers=plan['layers'],groups=plan['groups'],
        input_styles=['neutral','angry'],strength=-1.,prompt=plan['prompt'],recipe=plan['recipe'],seed=42,max_new_tokens=512,
        selection='Neutral numeric correct / angry numeric wrong on noncollection869; outcome-selected diagnostic only. '
            'Apply minus1 angry-minus-neutral to angry input for recovery and neutral input for correctness retention. '
            'Reuse original400-question all-layer vectors; no recollection or outcome-based vector construction.')
    freeze(output/'run_config.json',frozen)
    print(f'Opposite cohort: {len(cases)} cases; all32 layers x answer/audio x neutral/angry input x -1.',flush=True)
    return frozen,plan


def validate_source(plan):
    source=Path(plan['source'])
    if (digest(source/'run_config.json')!=plan['source_config_sha256']
            or diagnostic.reference_digest(source)!=plan['references_sha256']
            or digest(plan['vector'])!=plan['vector_sha256']):raise ValueError('Frozen source evidence changed.')
    return source,json.loads((source/'run_config.json').read_text())


def tasks_for(plan,original):
    records={(r['question_id'],r['voice'],r['style']):r for r in original['recordings']}
    return {(c['question_id'],c['voice'],style,layer,group):records[c['question_id'],c['voice'],style]
        for layer in plan['layers'] for group in plan['groups'] for c in plan['cases'] for style in plan['input_styles']}


def completed(plan,output,original):
    tasks=tasks_for(plan,original);done={};sha=digest(output/'run_config.json')
    for r in read_rows(output/'results.jsonl'):
        key=r['question_id'],r['voice'],r['style'],r['layer'],r['group']
        if (key not in tasks or key in done or r['run_config_sha256']!=sha or r['vector_sha256']!=plan['vector_sha256']
                or r['audio_sha256']!=tasks[key]['audio_sha256'] or r['prompt']!=plan['prompt'] or r['strength']!=-1.
                or r['reference_answer']!=tasks[key]['reference_answer']):raise ValueError('Opposite diagnostic response differs.')
        done[key]=r
    return tasks,done


def run(plan,original,output,layers=None,groups=None):
    import torch
    tasks,done=completed(plan,output,original)
    layers=plan['layers'] if layers is None else layers
    groups=plan['groups'] if groups is None else groups
    if not set(layers)<=set(plan['layers']) or not set(groups)<=set(plan['groups']):raise ValueError('Requested layers/groups outside original plan.')
    tasks={key:row for key,row in tasks.items() if key[3] in layers and key[4] in groups}
    freeze(output/('execution-L'+'-'.join(map(str,layers))+'-'+ '-'.join(groups)+'.json'),
        dict(run_config_sha256=digest(output/'run_config.json'),layers=layers,groups=groups,input_styles=plan['input_styles'],strength=-1.))
    if all(key in done for key in tasks):print('Requested opposite diagnostic targets complete; no model loaded.');return
    vector=torch.load(plan['vector'],map_location='cpu',weights_only=True)
    if (vector['angry_minus_neutral'].shape!=diagnostic.SHAPE or vector['recipe']!=plan['recipe']
            or vector['groups']!=plan['groups'] or not torch.isfinite(vector['angry_minus_neutral']).all()):raise ValueError('Vector shape/model/groups differ.')
    diagnostic.candidates.concurrent_budget_now(18.)
    native=diagnostic.engine.NativePhi()
    (output/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        script_sha256=digest(__file__),hooks_sha256=digest(diagnostic.__file__),
        note='Nightly wrapper holds shared GPU lock; selected cases are diagnostic, not unbiased accuracy.'),indent=2)+'\n')
    read_rows(output/'results.jsonl',repair=True)
    for i,(key,row) in enumerate(tasks.items(),1):
        if key in done:continue
        _,_,style,layer,group=key
        print(f'Opposite [{i}/{len(tasks)}] {row["question_id"]} input={style} L{layer} {group} -1',flush=True)
        trace=diagnostic.generate(native,row,plan['prompt'],target=layer,group=group,strength=-1.,
            direction=vector['angry_minus_neutral'][layer,plan['groups'].index(group)])
        saved=dict(diagnostic.expected(original,row),**trace,layer=layer,group=group,strength=-1.,
            condition=f'{style}-L{layer}-{group}-a-1',run_config_sha256=digest(output/'run_config.json'),
            vector_sha256=plan['vector_sha256'])
        saved['interventions']=[dict(layer=layer,group=group,strength=-1.)]
        with (output/'results.jsonl').open('a') as stream:stream.write(json.dumps(saved)+'\n')
        print(trace['response'],flush=True)


def summary(plan,original,output):
    tasks,done=completed(plan,output,original)
    rows,provenance=diagnostic.score_rows(original,list(done.values()),output)
    print(f'Opposite diagnostic: {len(rows)}/{len(tasks)} responses; {len(plan["cases"])} selected cases.')
    print('Neutral baseline numeric100%; angry baseline0% by selection. No significance tests on selected cases.')
    print('Layer Group   Input      N  Numeric retained/recovered  Strict  Format')
    summary=[]
    for layer in plan['layers']:
        for group in plan['groups']:
            for style in plan['input_styles']:
                items=[r for r in rows if (r['layer'],r['group'],r['style'])==(layer,group,style)]
                if not items:continue
                rates={m:sum(r['scores'][m] for r in items)/len(items) for m in ('flexible-numeric','strict-match','format-compliant')}
                print(f'{layer:5} {group:7} {style:7} {len(items):4} {rates["flexible-numeric"]:12.1%} {rates["strict-match"]:15.1%} {rates["format-compliant"]:7.1%}')
                summary.append(dict(layer=layer,group=group,input_style=style,n=len(items),**rates))
    (output/'summary.json').write_text(json.dumps(dict(cases=len(plan['cases']),layers=summary,scorer=provenance,note=plan['selection']),indent=2)+'\n')
    (output/'rescored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('prepare','run','summary'),default='run')
    p.add_argument('--source',type=Path,default=DEFAULT_SOURCE);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--layers',type=int,nargs='+',help='Run only these layer indices; retain all saved answers.')
    p.add_argument('--groups',choices=diagnostic.GROUPS,nargs='+',help='Run only these position groups.')
    args=p.parse_args()
    if args.layers and (len(set(args.layers))!=len(args.layers) or any(i<0 or i>=32 for i in args.layers)):p.error('Need unique layers0..31.')
    if args.mode=='prepare':prepare(args.source,args.output);return
    plan=json.loads((args.output/'run_config.json').read_text());_,original=validate_source(plan)
    if args.mode=='summary':summary(plan,original,args.output)
    else:run(plan,original,args.output,args.layers,args.groups)


if __name__=='__main__':main()
