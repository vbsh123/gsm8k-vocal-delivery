"""Full869 noncollection GSM8K questions, fixed L7 answer vector, strengths minus1/minus1.5."""
import argparse
import fcntl
import json
from pathlib import Path

import phi4_all_layer_diagnostic as diagnostic
import phi4_opposite_recovery as opposite
import steer_phi4_gsm_reverse as engine
from kimi_bbh_layer_diagnostic import atomic_save,digest,freeze
from steer import read_rows

DEFAULT_OUTPUT=Path('outputs/phi4-gsm400-test869-L7-minus1-minus1.5')


def result_files(output):
    return [output/'results.jsonl']+sorted(output.glob('results-worker*.jsonl'))


def completed(plan,output):
    return engine.completed(plan,output,digest(output/'run_config.json'),result_paths=result_files(output))


def fixed_vector(plan,output):
    import torch
    source=Path(plan['source_diagnostic'])
    if digest(source/'run_config.json')!=plan['source_config_sha256'] or digest(source/'style-vectors.pt')!=plan['source_vector_sha256']:
        raise ValueError('Original collection/vector changed.')
    if diagnostic.reference_digest(source)!=plan['references_sha256']:raise ValueError('Reference answers changed.')
    vector=torch.load(source/'style-vectors.pt',map_location='cpu',weights_only=True)
    if (vector['angry_minus_neutral'].shape!=diagnostic.SHAPE or vector['groups']!=['answer','audio']
            or vector['recipe']!=plan['recipe'] or vector['run_config_sha256']!=plan['source_config_sha256']
            or vector['collection_question_ids']!=[p['question_id'] for p in plan['collection']]
            or not torch.isfinite(vector['angry_minus_neutral']).all()):raise ValueError('All-layer vector identity/shape differs.')
    selected=vector['angry_minus_neutral'][7,0].clone()
    state=dict(protocol=plan['protocol'],run_config_sha256=digest(output/'run_config.json'),model=plan['recipe']['model'],
        revision=plan['recipe']['revision'],layer=7,group='answer',source_vector_sha256=plan['source_vector_sha256'],
        collection_question_ids=vector['collection_question_ids'],definition=plan['vector_definition'],angry_minus_neutral=selected)
    path=output/'style-vectors.pt'
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=True)
        if old['run_config_sha256']!=state['run_config_sha256'] or not torch.equal(old['angry_minus_neutral'],selected):
            raise ValueError('Saved L7 vector differs.')
    else:atomic_save(path,state)
    return selected


def append_reused(plan,output,row,condition,source):
    tasks={(r['question_id'],r['voice']):r for r in plan['test']};key=row['question_id'],row['voice']
    if key not in tasks:return
    task=tasks[key]
    if (row['audio_sha256']!=task['audio_sha256'] or row['prompt']!=plan['prompt'] or row['seed']!=42
            or row['max_new_tokens']!=512 or row['model']!=plan['recipe']['model'] or row['revision']!=plan['recipe']['revision']
            or row['reference_answer']!=task['reference_answer']):raise ValueError('Reused generation recipe/input differs.')
    strength=plan['strengths'][condition]
    saved=dict(task,response=row['response'],response_tokens=row['response_tokens'],condition=condition,protocol=plan['protocol'],
        run_config_sha256=digest(output/'run_config.json'),prompt=plan['prompt'],seed=42,max_new_tokens=512,
        model=plan['recipe']['model'],revision=plan['recipe']['revision'],strength=strength,layer=7,group='answer',
        vector_sha256=digest(output/'style-vectors.pt') if strength else None,
        interventions=[dict(layer=7,group='answer',strength=strength)] if strength else [],
        reused_evidence=dict(source=str(source),original_condition=row.get('condition',row['style']),
            original_vector_sha256=row.get('vector_sha256')))
    with (output/'results.jsonl').open('a') as stream:stream.write(json.dumps(saved)+'\n')


def reuse(plan,original,source,output):
    read_rows(output/'results.jsonl',repair=True)
    _,done=completed(plan,output)
    _,refs=diagnostic.references(original,source)
    for row in refs.values():
        key=row['question_id'],row['voice'],'neutral'
        if row['style']=='neutral' and key not in done:
            append_reused(plan,output,row,'neutral',source);done[key]=row
    sources=[]
    if (source/'challenge.json').exists():
        rows,_=diagnostic.completed(original,source,json.loads((source/'challenge.json').read_text()));sources.append((source,rows))
    opposite_folder=opposite.DEFAULT_OUTPUT
    if (opposite_folder/'run_config.json').exists():
        other=json.loads((opposite_folder/'run_config.json').read_text());opposite.validate_source(other)
        if Path(other['source']).resolve()!=source.resolve() or other['vector_sha256']!=plan['source_vector_sha256']:
            raise ValueError('Opposite diagnostic used another source vector.')
        _,rows=opposite.completed(other,opposite_folder,original);sources.append((opposite_folder,list(rows.values())))
    for folder,rows in sources:
        for row in rows:
            condition='neutral-L7-answer-a-1';key=row['question_id'],row['voice'],condition
            if row['layer']==7 and row['group']=='answer' and row['style']=='neutral' and row['strength']==-1. and key not in done:
                append_reused(plan,output,row,condition,folder);done[key]=row
    print(f'Reused {sum(k[2]=="neutral" for k in done)} neutral baselines and {sum(k[2]=="neutral-L7-answer-a-1" for k in done)} compatible L7 -1 answers.',flush=True)


def prepare(source,output):
    original=json.loads((source/'run_config.json').read_text());tasks,done=diagnostic.references(original,source)
    if len(done)!=len(tasks):raise ValueError('Finish original neutral/angry reference answers first.')
    tests=[r for r in original['recordings'] if r['style']=='neutral'];train={p['question_id'] for p in original['collection']}
    ids={r['question_id'] for r in tests}
    if len(train)!=400 or len(tests)!=869 or len(ids)!=869 or train&ids:raise ValueError('Expected400 collection and869 disjoint tests.')
    output.mkdir(parents=True,exist_ok=True)
    plan=dict(protocol='phi4-gsm400-test869-L7-minus1-minus1.5-v1',dataset='GSM8K',report_label='Phi-4 GSM8K L7 shortlist follow-up',
        source_diagnostic=str(source.absolute()),source_config_sha256=digest(source/'run_config.json'),
        source_vector_sha256=digest(source/'style-vectors.pt'),references_sha256=diagnostic.reference_digest(source),
        fixed_vector_source=str((source/'style-vectors.pt').absolute()),collection=original['collection'],test=tests,
        bank={r['question_id']:original['bank'][r['question_id']] for r in tests},recipe=original['recipe'],prompt=original['prompt'],seed=42,max_new_tokens=512,
        layer=7,group='answer',strengths={'neutral':0.,'neutral-L7-answer-a-1':-1.,'neutral-L7-answer-a-1.5':-1.5},
        vector_definition='Raw L7 answer mean angry-minus-neutral from original400 all-layer collection, unchanged/unscaled.',
        scoring='Installed upstream GSM8K filters and metric; numeric equivalence and format diagnostic additional. '
            'Native audio adaptation. Full869 noncollection cohort; L7 chosen after inspecting recovery/retention on subsets '
            'of these same869. Exploratory layer/strength follow-up, not untouched confirmation; tests/intervals unadjusted.')
    freeze(output/'run_config.json',plan);fixed_vector(plan,output);reuse(plan,original,source,output)
    print('Ready869 questions; neutral baseline reused; L7 answer -1 and -1.5 only. No new vector collection or TTS.',flush=True)
    return plan


def execute(plan,output,worker_id=None):
    vector=fixed_vector(plan,output)
    _,done=completed(plan,output)
    conditions=list(plan['strengths']) if worker_id is None else [list(plan['strengths'])[worker_id+1]]
    needed=[(r['question_id'],r['voice'],c) for r in plan['test'] for c in conditions]
    if all(k in done for k in needed):print('Assigned answers complete; no model loaded.');return
    import torch
    diagnostic.candidates.concurrent_budget_now(18.)
    native=engine.NativePhi(7)
    suffix='' if worker_id is None else f'-worker{worker_id}'
    (output/f'runtime{suffix}.json').write_text(json.dumps(dict(torch=torch.__version__,transformers=engine.version('transformers'),
        gpu=torch.cuda.get_device_name(0),script_sha256=digest(__file__),engine_sha256=digest(engine.__file__),
        worker_id=worker_id,conditions=conditions,gpu_budget_gib=18.),indent=2)+'\n')
    engine.evaluate(plan,output,native,vector,digest(output/'run_config.json'),conditions=conditions,
        result_file=output/('results.jsonl' if worker_id is None else f'results-worker{worker_id}.jsonl'),
        result_paths=result_files(output))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('prepare','run','worker','summary'),default='run')
    p.add_argument('--source',type=Path,default=diagnostic_source());p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--worker-id',type=int,choices=(0,1));p.add_argument('--strength',type=float,choices=(-1.,-1.5),help='Optional individual-strength summary.')
    args=p.parse_args()
    if args.mode=='summary':
        conditions=None if args.strength is None else ['neutral',f'neutral-L7-answer-a{args.strength:g}']
        engine.report(args.output,result_paths=result_files(args.output),conditions=conditions,
            report_suffix='' if args.strength is None else f'-strength{args.strength:g}');return
    if args.mode=='prepare':prepare(args.source,args.output);return
    if args.mode=='worker' and args.worker_id is None:p.error('Worker needs --worker-id0 or1.')
    plan=json.loads((args.output/'run_config.json').read_text())
    if args.mode=='run':execute(plan,args.output);return
    with (args.output/f'worker{args.worker_id}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(plan,args.output,args.worker_id)


def diagnostic_source():return Path('outputs/phi4-gsm400-all-layers-test869-minus1')


if __name__=='__main__':main()
