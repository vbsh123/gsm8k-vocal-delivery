"""Portable angry/agreed active-pooling fit200/eval200 experiment for the Phi host."""
import argparse
from copy import deepcopy
import fcntl
from importlib.metadata import version
import json
import io
from pathlib import Path
import random
import subprocess
import time
import zipfile

import torch
import diffa_active_pooling as pooling
from diffa_model_receipts import model_revisions

probe=pooling.probe
vanilla=pooling.vanilla
winners=pooling.winners
MARKER='DIFFA_ACTIVE_POOLING_INPUTS.json'


def package(source, old_bundle, output, questions=200):
    parent=json.loads((source/'run_config.json').read_text())
    job,payload=pooling.bundle_data(old_bundle)
    checkpoint=torch.load(source/'collection-stream.pt',map_location='cpu',weights_only=True)
    split=json.loads((source.parent.parent/'split.json').read_text())
    if (parent['emotion']!='angry' or parent['regime']!='agreed' or parent['prompt']!=vanilla.PROMPT
            or job['emotion']!='angry' or job['regime']!='agreed'
            or parent['split_sha256']!=vanilla.fingerprint(split)
            or checkpoint['run_config_sha256']!=winners.digest(source/'run_config.json')
            or checkpoint['completed_ids']!=parent['collection_ids']
            or winners.digest(source/'style-vectors.pt')!=job['source']['vector_sha256']
            or probe.sha(payload['style-vectors.pt'])!=job['source']['vector_sha256']
            or parent['prompt']!=job['target']['prompt']
            or parent['max_new_tokens']!=job['target']['max_new_tokens']):
        raise ValueError('Source collection, classified split, and evaluation bundle differ.')
    if not 1<=questions<=len(parent['collection_ids']):
        raise ValueError('Invalid fitting question count.')
    keep=set(random.Random(42).sample(sorted(parent['collection_ids']),questions))
    pairs=[p for p in parent['pairs'] if p['question_id'] in keep]
    labels={}
    for pair in pairs:
        q=pair['question_id']
        if (split['emotion2vec_predictions'][q]['angry']!='angry'
                or split['audio_sha256'][q]['angry']!=pair['emotion']['audio_sha256']
                or split['audio_sha256'][q]['neutral']!=pair['neutral']['audio_sha256']):
            raise ValueError(f'Angry classification/audio provenance differs: {q}')
        labels[q]=dict(emotion='angry',audio_sha256=pair['emotion']['audio_sha256'])
    if len(pairs)!=questions or keep.intersection(r['question_id'] for r in job['target']['records']):
        raise ValueError('Fitting pairs incomplete or overlap evaluation.')
    ids=[p['question_id'] for p in pairs]
    fit=deepcopy(parent)
    fit.update(protocol='diffa-active-pooling-portable-fit200-v1',pairs=pairs,collection_ids=ids,
               parent_collection_config_sha256=winners.digest(source/'run_config.json'),
               selection='Seed42 uniform sample from original angry/agreed fitting IDs; no outcome filtering.')
    old_rows={r['question_id']:r for r in checkpoint['rows']}
    expected={q:{k:old_rows[q][k] for k in ('neutral_response','emotion_response','neutral_forwards','emotion_forwards','matched_forwards')} for q in ids}
    baseline=pooling.read_rows_from_bytes(payload['results.jsonl'])
    with zipfile.ZipFile(old_bundle) as archive:
        environment=json.loads(archive.read(probe.MARKER))
    data=dict(protocol='diffa-active-pooling-portable-inputs-v1',fit=fit,job=job,baseline=baseline,
              source_rows=expected,agreed_labels=labels,models=environment['models'],packages=environment['packages'],
              parent_evaluation_bundle_sha256=winners.digest(old_bundle),
              note='New full-buffer and active-block vectors both fit the SAME selected200 pairs. '
                   'Original425 mean cannot be reconstructed for a subset from aggregate checkpoints. '
                   'Evaluation uses the exact original200 questions, separate from fitting.')
    encoded=probe.encoded(data)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        old=load_bundle(output)
        if old!=data:
            raise ValueError('Existing portable bundle differs; choose another output.')
    else:
        temporary=output.with_suffix('.zip.tmp')
        try:
            with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
                archive.writestr('inputs.json',encoded)
                archive.writestr(MARKER,probe.encoded(dict(sha256=probe.sha(encoded))))
            temporary.rename(output)
        finally:
            temporary.unlink(missing_ok=True)
    output.with_suffix('.zip.sha256').write_text(f'{winners.digest(output)}  {output.name}\n')
    print(f'DOWNLOAD: {output.resolve()}\nSIZE: {output.stat().st_size/2**20:.2f} MiB. '
          f'{len(ids)} recognized angry fitting questions; {len(baseline)} unchanged, disjoint evaluation questions. '
          'Includes metadata and replay responses; no audio or model weights.',flush=True)
    return output


def load_bundle(path):
    with zipfile.ZipFile(path) as archive:
        data=archive.read('inputs.json')
        if probe.sha(data)!=json.loads(archive.read(MARKER))['sha256']:
            raise ValueError('Portable input checksum differs.')
    return json.loads(data)


def from_existing_probe(bundle,registry,questions=200):
    """Resolve fitting records from the already-extracted Jenny metadata, not another upload."""
    job,payload=pooling.bundle_data(bundle)
    if (job['emotion']!='angry' or job['regime']!='agreed' or job['target']['prompt']!=vanilla.PROMPT
            or probe.sha(payload['style-vectors.pt'])!=job['source']['vector_sha256']):
        raise ValueError('Expected the original vanilla angry/agreed layer-probe bundle.')
    _,info=winners.read_vector(io.BytesIO(payload['style-vectors.pt']),'angry',31,'answer')
    if info['collection_prompt']!=job['target']['prompt'] or info['collection_tokens']!=job['target']['max_new_tokens']:
        raise ValueError('Original vector collection prompt or budget differs.')
    original_ids=info['extraction_ids']
    if len(set(original_ids))!=len(original_ids) or not 1<=questions<=len(original_ids):
        raise ValueError('Original fitting IDs incomplete or requested count too large.')
    keep=set(random.Random(42).sample(sorted(original_ids),questions))
    ids=[q for q in original_ids if q in keep]
    if set(ids).intersection(r['question_id'] for r in job['target']['records']):
        raise ValueError('Fitting questions overlap evaluation.')
    pairs=[]
    labels={}
    bank=deepcopy(job['target']['bank'])
    for q in ids:
        neutral=registry.choose('gsm8k',q,'neutral')
        if neutral is None:
            raise ValueError(f'Jenny metadata lacks neutral fitting question {q}.')
        angry=registry.choose('gsm8k',q,'angry',question=neutral['question'],answer=neutral['answer'],agreed=True)
        if angry is None:
            raise ValueError(f'Jenny metadata lacks the e2v-agreed angry fitting recording for {q}.')
        for style,item in [('neutral',neutral),('angry',angry)]:
            variants={r['record']['audio_sha256'] for r in registry.options['gsm8k',q,style]
                      if r['question']==neutral['question'] and r['answer']==neutral['answer']}
            if len(variants)!=1:
                raise ValueError(f'Ambiguous Jenny recordings: {q}/{style}.')
        pairs.append(dict(question_id=q,neutral=neutral['record'],emotion=angry['record']))
        labels[q]=dict(emotion='angry',audio_sha256=angry['record']['audio_sha256'])
        bank[q]=dict(question_id=q,question=neutral['question'],answer=neutral['answer'])
    baseline=pooling.read_rows_from_bytes(payload['results.jsonl'])
    fit=dict(protocol='diffa-active-pooling-portable-fit200-v1',name='gsm8k',emotion='angry',regime='agreed',
        prompt=job['target']['prompt'],collection_prompt=job['target']['prompt'],
        **{k:job['target'][k] for k in vanilla.RECIPE},
        bank=bank,target=deepcopy(job['target']),collection_ids=ids,pairs=pairs,
        test_ids=[r['question_id'] for r in job['target']['records']],
        selection='Seed42 uniform sample from original angry/agreed vector fitting IDs; reverify stored e2v labels by audio hash.',
        parent_source_vector_sha256=job['source']['vector_sha256'])
    with zipfile.ZipFile(bundle) as archive:
        environment=json.loads(archive.read(probe.MARKER))
    return dict(protocol='diffa-active-pooling-portable-inputs-v1',fit=fit,job=job,baseline=baseline,
        source_rows={q:None for q in ids},agreed_labels=labels,models=environment['models'],packages=environment['packages'],
        parent_evaluation_bundle_sha256=winners.digest(bundle),construction='existing-probe-and-jenny-v1',
        note='Fitting metadata and hash-verified e2v labels resolved from existing Jenny ZIP. '
             'Old fitting responses were not bundled; fresh responses are retained. '
             'Exact evaluation neutral replay is checked before collection.')


def input_data(bundle,roots,output,summary_only=False):
    with zipfile.ZipFile(bundle) as archive:
        markers=set(archive.namelist())
    if MARKER in markers:
        return load_bundle(bundle)
    if probe.MARKER not in markers:
        raise ValueError('Expected a DIFFA layer-probe ZIP or active-pooling input ZIP.')
    cached=output/'local-inputs.json'
    if cached.exists():
        saved=json.loads(cached.read_text())
        data=saved['data']
        if saved['data_sha256']!=vanilla.fingerprint(data) or data['parent_evaluation_bundle_sha256']!=winners.digest(bundle):
            raise ValueError('Locally resolved inputs differ from the frozen probe bundle.')
        return data
    if summary_only:
        return None
    root=next((Path(r) for r in roots if (Path(r)/'metadata/audio_inventory.jsonl').is_file()
               or (Path(r)/'outputs/shared-three-emotions/audio_inventory.jsonl').is_file()),None)
    if root is None:
        raise ValueError('Locate the extracted Jenny audio directory with --audio-root (normally outputs/jenny-zip-audio).')
    print(f'Using existing layer ZIP and Jenny metadata at {root}; no additional ZIP needed.',flush=True)
    registry=vanilla.Registry(root)
    data=from_existing_probe(bundle,registry)
    winners.freeze(cached,dict(data=data,data_sha256=vanilla.fingerprint(data)))
    print(f'Resolved {len(data["fit"]["pairs"])} angry/agreed fitting pairs and {len(data["baseline"])} disjoint evaluation questions.',flush=True)
    return data


def prepare(data,bundle,output):
    fit=data['fit']
    ids=fit['collection_ids']
    baseline=data['baseline']
    job=deepcopy(data['job'])
    expected={r['question_id']:r for r in job['target']['records']}
    if (data['protocol']!='diffa-active-pooling-portable-inputs-v1'
            or fit['emotion']!='angry' or fit['regime']!='agreed' or fit['prompt']!=vanilla.PROMPT
            or fit['prompt']!=job['target']['prompt']
            or fit['max_new_tokens']!=job['target']['max_new_tokens']
            or len(ids)!=len(set(ids)) or [p['question_id'] for p in fit['pairs']]!=ids
            or set(data['source_rows'])!=set(ids) or set(ids).intersection(expected)
            or len(baseline)!=len(expected) or {r['question_id'] for r in baseline}!=set(expected)):
        raise ValueError('Portable fitting/evaluation cohorts or recipe differ.')
    if any(value is None for value in data['source_rows'].values()) and data.get('construction')!='existing-probe-and-jenny-v1':
        raise ValueError('Missing source replay metadata without local Jenny reconstruction.')
    for pair in fit['pairs']:
        label=data['agreed_labels'][pair['question_id']]
        if label!=dict(emotion='angry',audio_sha256=pair['emotion']['audio_sha256']):
            raise ValueError('Portable fitting classification differs.')
    for row in baseline:
        if (row.get('strength')!=0. or row.get('interventions')!=[]
                or not winners.reusable(row,{},expected[row['question_id']],job['target'])):
            raise ValueError('Portable baseline recipe differs.')
    output.mkdir(parents=True,exist_ok=True)
    receipt=dict(protocol=data['protocol'],input_bundle_sha256=winners.digest(bundle),layers=[31],
        scopes=list(pooling.SCOPES),collection_ids=ids,evaluation_ids=list(expected),
        collection_count=len(ids),evaluation_count=len(expected),models=data['models'],packages=data['packages'],
        time_average='Matched forwards, unchanged.',
        answer_pooling='full-buffer512 versus current-block32; same generations for both.',
        steering='L31 answer x+1; inject across full answer buffer, unchanged.',
        runner_sha256=winners.digest(__file__),collector_sha256=winners.digest(pooling.__file__))
    # Steering-policy updates do not invalidate an unchanged collection checkpoint.
    # Keep the original runner hash as provenance; record the current execution separately.
    receipt_path=output/'experiment.json'
    if receipt_path.exists():
        original_receipt=json.loads(receipt_path.read_text())
        scientific=lambda value:{k:v for k,v in value.items() if k!='runner_sha256'}
        if scientific(original_receipt)!=scientific(receipt):
            raise ValueError('Frozen collection settings differ.')
    else:
        winners.freeze(receipt_path,receipt)
    winners.atomic_text(output/'steering-policy.json',json.dumps(dict(
        runner_sha256=winners.digest(__file__),steer_scopes=['active-block'],
        note='Keep both collected averages; generate steering responses only for active-block.'),indent=2)+'\n')
    plans={}
    for scope in pooling.SCOPES:
        plan=deepcopy(fit)
        plan.update(protocol=data['protocol']+'-'+scope,pooling_scope=scope,
                    averaging=f'{scope} answer mean; matched forwards; equal question weight.')
        winners.freeze(output/scope/'run_config.json',plan)
        plans[scope]=plan
    # Every new direction is fitted on200, not the original425: update provenance too.
    for key in ('source','target'):
        job[key]['extraction_ids']=ids
        job[key]['vector_info']=dict(job[key].get('vector_info',{}),extraction_ids=ids,pairs=len(ids))
    return plans,job


def collect(data,plans,output,paths,model,tokenizer,features):
    import src.fast_dllm.diffa_generate as decoder
    state=pooling.load_state(output,plans)
    fit=data['fit']
    for pair in fit['pairs'][len(state['scopes']['full-buffer']['completed_ids']):]:
        q=pair['question_id']
        print(f'Collect [{len(state["scopes"]["full-buffer"]["completed_ids"])+1}/{len(fit["pairs"])}] {q}: neutral + angry; both pooling scopes.',flush=True)
        traces=[]
        for style in ('neutral','emotion'):
            original=pair[style]
            record=dict(original,audio_file=paths[original['audio_sha256']])
            traces.append(pooling.capture_poolings(model,
                lambda:vanilla.task.generate(record,fit,fit['prompt'],model,tokenizer,features),decoder,fit['max_new_tokens']))
        (nr,ne,n,na),(er,ee,e,ea)=traces
        expected=data['source_rows'][q]
        if expected is not None and (nr!=expected['neutral_response'] or er!=expected['emotion_response']
                or n.shape[1]!=expected['neutral_forwards'] or e.shape[1]!=expected['emotion_forwards']):
            raise ValueError(f'{q}: source response/forward replay differs; no steering started.')
        pooled,matched=pooling.pool_pair(n,e,na,ea)
        for scope in pooling.SCOPES:
            nm,em,delta=pooled[scope]
            pooling.running.append(state['scopes'][scope],dict(question_id=q,neutral=nm,emotion=em,difference=delta,
                neutral_response=nr,emotion_response=er,neutral_forwards=n.shape[1],emotion_forwards=e.shape[1],
                matched_forwards=matched,neutral_encoding=ne,emotion_encoding=ee))
        vanilla.task.save_tensor(torch,state,output/'collection-checkpoint.pt')
        del traces,n,e,na,ea,pooled
    for scope in pooling.SCOPES:
        pooling.running.finalize(plans[scope],output/scope,state['scopes'][scope])


def relocate(data,roots):
    records=[p[style] for p in data['fit']['pairs'] for style in ('neutral','emotion')]
    records+=data['job']['target']['records']
    return probe.resolve_audio(dict(target=dict(records=records)),roots)


def active_steering_jobs(plans,job,data,output,create=True):
    return [(child,selected) for child,selected in
            pooling.steering_jobs(plans,job,data['baseline'],output,[31],create=create)
            if selected['pooling_scope']=='active-block']


def summary(plans,job,data,output):
    state=pooling.load_state(output,plans)
    print(f'Angry/agreed fitting: {len(state["scopes"]["full-buffer"]["completed_ids"])}/{len(data["fit"]["pairs"])}; '
          f'evaluation: {len(data["baseline"])} separate questions; L31 answer x+1.',flush=True)
    for child,selected in active_steering_jobs(plans,job,data,output,create=False):
        print(f'{selected["pooling_scope"]} / L31',flush=True)
        vanilla.task.report_steering(child)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('pack','run','summary'),default='run')
    parser.add_argument('--source',type=Path,default=Path('outputs/diffa-gsm-vanilla/angry/agreed'))
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--audio-root',type=Path,action='append')
    args=parser.parse_args()
    if args.mode=='pack':
        package(args.source,args.bundle,args.output or Path('outputs/gsm8k-angry-agreed-active-fit200-eval200.zip'))
        return
    output=args.output or Path('outputs/diffa-active-pooling-fit200-eval200')
    repo=Path(__file__).resolve().parent
    roots=[*(args.audio_root or [repo/'outputs/jenny-zip-audio']),repo]
    data=input_data(args.bundle,roots,output,summary_only=args.mode=='summary')
    if data is None:
        print('Active-pooling experiment not started yet. Run with this same layer-probe ZIP to collect.',flush=True)
        return
    plans,job=prepare(data,args.bundle,output)
    if args.mode=='summary':
        summary(plans,job,data,output)
        return
    with (output/'runner.lock').open('a') as own:
        try:
            fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('This active-pooling experiment is already running.') from None
        repo=Path(__file__).resolve().parent
        print('Checking exact audio hashes for fitting and evaluation; CPU only.',flush=True)
        paths,mappings=relocate(data,[*(args.audio_root or [repo/'outputs/jenny-zip-audio']),repo])
        winners.atomic_text(output/'audio-relocations.json',json.dumps(mappings,indent=2)+'\n')
        print(f'Verified {len(mappings)} exact audio hashes; Phi recordings/configs unchanged.',flush=True)
        for name in probe.PACKAGES:
            if version(name)!=data['packages'][name]:
                raise ValueError(f'{name} differs from source environment.')
        if model_revisions()!=data['models']:
            raise ValueError('Pinned model revisions differ from source.')
        with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
            print('Waiting for shared GPU lock and idle GPU.',flush=True)
            fcntl.flock(gpu,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                time.sleep(30)
            from extract_vectors import load_model
            model,tokenizer,features=load_model()
            winners.atomic_text(output/'runtime.json',json.dumps(dict(models=model_revisions(),
                packages={p:version(p) for p in probe.PACKAGES},model_config=model.config.to_dict()),indent=2,default=str)+'\n')
            # Check the exact evaluation baseline before spending time on fresh collection.
            for row in random.Random(42).sample(data['baseline'],min(3,len(data['baseline']))):
                record=dict(row,audio_file=paths[row['audio_sha256']])
                response,_=vanilla.task.generate(record,job['target'],job['target']['prompt'],model,tokenizer,features)
                if response!=row['response']:
                    raise ValueError('Evaluation neutral replay differs; collection not started.')
            collect(data,plans,output,paths,model,tokenizer,features)
            for child,selected in active_steering_jobs(plans,job,data,output):
                runtime=probe.with_audio_paths(selected,paths)
                probe.validate(runtime,child,check_audio=True)
                winners.generate_job(runtime,child,model,tokenizer,features)
                probe.validate(runtime,child)
                print(f'{selected["pooling_scope"]} / L31',flush=True)
                vanilla.task.report_steering(child,tokenizer)
            print('DONE: both200-pair averages collected; ONLY active-block steered at L31 x+1 on the disjoint200 questions.',flush=True)


if __name__=='__main__':
    main()
