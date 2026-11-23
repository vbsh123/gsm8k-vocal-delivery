"""Compare full-buffer and current-block collection on frozen vanilla angry/agreed pairs."""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import subprocess
import time
import zipfile

import torch
import diffa_layer_probe as probe
import diffa_stream_collection as running
from steer import diffusion_progress, read_rows

vanilla = probe.vanilla
winners = vanilla.task.winners
SCOPES = ('full-buffer', 'active-block')


def capture_poolings(model, generate, decoder, budget=512, block_length=32):
    """Observe both scopes in the SAME generation; never modify model states."""
    layers = model.llm_model.model.transformer.blocks
    active = [[] for _ in layers]
    handles = []
    with diffusion_progress(model, budget, block_length) as progress:
        def hook(index):
            def observe(module, inputs, output):
                hidden = output[0]
                if 'block' not in progress or hidden.shape[1] < budget:
                    raise ValueError('Missing full answer-buffer or native decoder block state.')
                start = hidden.shape[1]-budget+progress['block']*block_length
                stop = min(start+block_length, hidden.shape[1])
                if not hidden.shape[1]-budget <= start < stop <= hidden.shape[1]:
                    raise ValueError('Invalid current answer-block bounds.')
                active[index].append(hidden[0,start:stop].detach().float().mean(0).cpu())
            return observe
        try:
            handles = [block.register_forward_hook(hook(i)) for i,block in enumerate(layers)]
            response, encoding, full = vanilla.task.emotion.capture(model,generate,decoder,budget)
        finally:
            for handle in handles:
                handle.remove()
    active = torch.stack([torch.stack(values) for values in active])
    if active.shape[:2] != full.shape[:2]:
        raise ValueError('Active/full captures have different forwards.')
    return response, encoding, full, active


def pool_pair(neutral, emotion, neutral_active, emotion_active):
    nm, em, delta, matched = vanilla.task.average_pair(neutral,emotion,'matched')
    na, ea, da, active_matched = vanilla.task.average_pair(neutral_active.unsqueeze(2),emotion_active.unsqueeze(2),'matched')
    if matched != active_matched:
        raise ValueError('Different time averaging between pooling scopes.')
    result = {'full-buffer': (nm,em,delta)}
    active = [value.clone() for value in (nm,em,delta)]
    for value,answer in zip(active,(na,ea,da)):
        value[:,3] = answer[:,0]
    result['active-block'] = tuple(active)
    return result, matched


def bundle_data(path):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read(probe.MARKER))
        prefix = manifest['folder']
        payload = {}
        for suffix in ('run_config.json','results.jsonl','style-vectors.pt'):
            name = prefix+'/'+suffix
            data = archive.read(name)
            if probe.sha(data) != manifest['files'][name]:
                raise ValueError(f'Bundle hash differs: {name}')
            payload[suffix] = data
    return json.loads(payload['run_config.json']), payload


def prepare(source, bundle, output, layers):
    parent = json.loads((source/'run_config.json').read_text())
    if parent['emotion'] != 'angry' or parent['regime'] != 'agreed' or parent['prompt'] != vanilla.PROMPT:
        raise ValueError('Use the frozen vanilla angry/agreed source.')
    vector = source/'style-vectors.pt'
    job, payload = bundle_data(bundle)
    ids = parent['collection_ids']
    if (job['emotion'] != 'angry' or job['regime'] != 'agreed'
            or winners.digest(vector) != job['source']['vector_sha256']
            or probe.sha(payload['style-vectors.pt']) != winners.digest(vector)
            or job['target']['prompt'] != parent['prompt']
            or job['target']['max_new_tokens'] != parent['max_new_tokens']
            or set(ids).intersection(r['question_id'] for r in job['target']['records'])):
        raise ValueError('Bundle does not use the exact source vector, prompt, budget, and held-out questions.')
    baseline = read_rows_from_bytes(payload['results.jsonl'])
    expected = {r['question_id']:r for r in job['target']['records']}
    if len(baseline) != len(expected) or {r['question_id'] for r in baseline} != set(expected):
        raise ValueError('Bundle neutral baseline incomplete/duplicate.')
    for row in baseline:
        if (row.get('strength') != 0. or row.get('interventions') != []
                or not winners.reusable(row,{},expected[row['question_id']],job['target'])):
            raise ValueError('Bundle baseline has a different generation recipe.')
    output.mkdir(parents=True,exist_ok=True)
    receipt = dict(protocol='diffa-current-block-pooling-ablation-v1',
        source_config_sha256=winners.digest(source/'run_config.json'),
        source_vector_sha256=winners.digest(vector),bundle_sha256=winners.digest(bundle),
        layers=layers,scopes=list(SCOPES),collection_ids=ids,
        evaluation_ids=[r['question_id'] for r in job['target']['records']],
        temporal_average='Matched chronological forward prefixes, unchanged from parent.',
        active_scope='Current 32-token answer block, including committed and still-masked positions.',
        intervention_scope='Full answer buffer, unchanged from parent; x+1.',
        collector_sha256=winners.digest(__file__))
    winners.freeze(output/'experiment.json',receipt)
    plans = {}
    for scope in SCOPES:
        plan = deepcopy(parent)
        plan.update(protocol=receipt['protocol']+'-'+scope,pooling_scope=scope,
            averaging=f'{scope} answer-position mean; matched forwards; equal question weight.')
        folder = output/scope
        winners.freeze(folder/'run_config.json',plan)
        plans[scope] = plan
    return parent, plans, job, baseline


def read_rows_from_bytes(data):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def load_state(output, plans):
    path = output/'collection-checkpoint.pt'
    checksum = winners.digest(output/'experiment.json')
    if path.exists():
        state = torch.load(path,map_location='cpu',weights_only=True)
        if state['experiment_sha256'] != checksum:
            raise ValueError('Checkpoint differs from the frozen experiment.')
    else:
        state = dict(experiment_sha256=checksum,scopes={s:dict(completed_ids=[],rows=[],
            difference_sum=None,unit_sum=None,valid_count=None) for s in SCOPES})
    prefixes = []
    for scope in SCOPES:
        saved = state['scopes'][scope]
        prefix = plans[scope]['collection_ids'][:len(saved['completed_ids'])]
        if saved['completed_ids'] != prefix or [r['question_id'] for r in saved['rows']] != prefix:
            raise ValueError('Collection checkpoint is not the exact fitting prefix.')
        prefixes.append(prefix)
    if prefixes[0] != prefixes[1]:
        raise ValueError('Pooling scopes have inconsistent checkpoints.')
    return state


def collect(parent, plans, source, output, model, tokenizer, features):
    import src.fast_dllm.diffa_generate as decoder
    state = load_state(output,plans)
    original = torch.load(source/'collection-stream.pt',map_location='cpu',weights_only=True)
    expected = {r['question_id']:r for r in original['rows']}
    if original['completed_ids'] != parent['collection_ids']:
        raise ValueError('Complete the original source collection first.')
    for pair in parent['pairs'][len(state['scopes']['full-buffer']['completed_ids']):]:
        q = pair['question_id']
        print(f'Collect [{len(state["scopes"]["full-buffer"]["completed_ids"])+1}/{len(parent["pairs"])}] {q}: both pooling scopes in one neutral/angry pass.',flush=True)
        traces = []
        for style in ('neutral','emotion'):
            record = pair[style]
            traces.append(capture_poolings(model,
                lambda: vanilla.task.generate(record,parent,parent['prompt'],model,tokenizer,features),
                decoder,parent['max_new_tokens']))
        (nr,ne,n,na),(er,ee,e,ea) = traces
        if nr != expected[q]['neutral_response'] or er != expected[q]['emotion_response']:
            raise ValueError(f'{q}: replay differs from original natural answers; no steering started.')
        pooled,matched = pool_pair(n,e,na,ea)
        for scope in SCOPES:
            nm,em,delta = pooled[scope]
            running.append(state['scopes'][scope],dict(question_id=q,neutral=nm,emotion=em,difference=delta,
                neutral_response=nr,emotion_response=er,neutral_forwards=n.shape[1],emotion_forwards=e.shape[1],
                matched_forwards=matched,neutral_encoding=ne,emotion_encoding=ee))
        vanilla.task.save_tensor(torch,state,output/'collection-checkpoint.pt')
        del traces,n,e,na,ea,pooled
    # The unchanged pooling control must reproduce the saved source direction before interventions.
    old = original['difference_sum'].float()/len(parent['pairs'])
    new = state['scopes']['full-buffer']['difference_sum'].float()/len(parent['pairs'])
    if not torch.allclose(old,new,rtol=1e-5,atol=1e-6):
        raise ValueError('Full-buffer control differs from original vector; no steering started.')
    for scope in SCOPES:
        running.finalize(plans[scope],output/scope,state['scopes'][scope])
    print('PASS: full-buffer control reproduces original direction; active-block vector uses the exact same pairs and forwards.',flush=True)


def steering_jobs(plans, parent_job, baseline, output, layers, create=True):
    result = []
    for scope in SCOPES:
        folder = output/scope
        if not (folder/'style-vectors.pt').exists():
            print(f'{scope}: vector collection not complete.',flush=True)
            continue
        for layer in layers:
            job = deepcopy(parent_job)
            job.update(protocol='diffa-current-block-pooling-steering-v1',layer=layer,group='answer',strengths=[0.,1.],pooling_scope=scope)
            for key in ('source','target'):
                job[key].update(vector=str((folder/'style-vectors.pt').resolve()),
                    vector_sha256=winners.digest(folder/'style-vectors.pt'))
            child = folder/f'L{layer}-answer-a1'
            config = child/'run_config.json'
            if not create:
                if not config.exists():
                    print(f'{scope}/L{layer}: steering not started.',flush=True)
                    continue
                if json.loads(config.read_text()) != job:
                    raise ValueError('Saved steering plan differs.')
            else:
                winners.freeze(config,job)
                results = child/'results.jsonl'
                if not results.exists():
                    with results.open('x') as stream:
                        for row in baseline:
                            stream.write(json.dumps(row)+'\n')
            probe.validate(job,child)
            result.append((child,job))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','run','summary'),default='run')
    parser.add_argument('--source',type=Path,default=Path('outputs/diffa-gsm-vanilla/angry/agreed'))
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/diffa-angry-agreed-active-pooling'))
    parser.add_argument('--layers',type=int,nargs='+',default=[31])
    args=parser.parse_args()
    layers=sorted(set(args.layers))
    if not layers or any(layer not in range(32) for layer in layers):
        parser.error('Layers must be 0..31.')
    if args.mode=='summary' and not (args.output/'experiment.json').exists():
        print('Active-pooling experiment not prepared yet.',flush=True)
        return
    parent,plans,job,baseline=prepare(args.source,args.bundle,args.output,layers)
    if args.mode=='prepare':
        print(f'Prepared {len(parent["pairs"])} exact fitting pairs; {len(baseline)} identical held-out questions; layers={layers}; strength=+1. Fresh activation collection required, no TTS.',flush=True)
        return
    if args.mode=='summary':
        state=load_state(args.output,plans)
        print(f'Paired collection: {len(state["scopes"]["full-buffer"]["completed_ids"])}/{len(parent["pairs"])}.',flush=True)
        for child,selected in steering_jobs(plans,job,baseline,args.output,layers,create=False):
            print(f'{selected["pooling_scope"]} / L{selected["layer"]}',flush=True)
            vanilla.task.report_steering(child)
        return
    with (args.output/'runner.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('Waiting for shared DIFFA GPU lock and idle GPU.',flush=True)
        with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
            fcntl.flock(gpu,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                time.sleep(30)
            from extract_vectors import load_model
            model,tokenizer,features=load_model()
            collect(parent,plans,args.source,args.output,model,tokenizer,features)
            for child,selected in steering_jobs(plans,job,baseline,args.output,layers):
                probe.validate(selected,child,check_audio=True)
                winners.generate_job(selected,child,model,tokenizer,features)
                probe.validate(selected,child)
                print(f'{selected["pooling_scope"]} / L{selected["layer"]}',flush=True)
                vanilla.task.report_steering(child,tokenizer)


if __name__=='__main__':
    main()
