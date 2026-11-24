"""Re-evaluate and refit all BBH emotions under original prompts, preserving night cohorts."""
import argparse
from copy import deepcopy
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import time

import emotion_block_night as night
from diffa_model_receipts import model_revisions

winners=night.winners
PROTOCOL='bbh-original-prompt-block-agreed-all-emotions-v1'


def original_prompt(name):
    if name not in ('bbh3','bbh5'):
        raise ValueError('This comparison is BBH3/BBH5 only.')
    count,choices,options=('three','A, B, C','A, B, or C') if name=='bbh3' else ('five','A, B, C, D, E','A, B, C, D, or E')
    return ('A logical deduction task which requires deducing the order of a sequence of objects. '
            f'Solve the problem in the audio, including its {count} choices {choices}. '
            "Let's think step by step. End with exactly one plain-text line in this format: "
            f'The answer is (<letter>). Replace <letter> with the correct option {options}.')


def prepare(name,source,output):
    parent=source/'diffa'/name
    config=parent/'run_config.json'
    old=json.loads(config.read_text())
    manifest=json.loads((parent/'cohort.json').read_text())
    if (old['protocol']!=night.PROTOCOL or old['model']!='diffa' or old['name']!=name
            or old['prompt']!=night.catalog.prompt_for(name)
            or old['collection']!='Current32-token answer-block means, matched chronological forward prefixes, equal question weight.'
            or night.vanilla.fingerprint(manifest)!=old['cohort_sha256']):
        raise ValueError('Need the exact completed bare-prompt overnight BBH cohort.')
    if len(night.raw_rows(parent,old))!=len(old['records']):
        raise ValueError('Complete the bare-prompt natural deliveries before this comparison.')
    new_manifest=deepcopy(manifest)
    new_manifest.update(protocol=PROTOCOL,prompt=original_prompt(name),parent_cohort_sha256=old['cohort_sha256'],
                        comparison='Same audio, questions, e2v fitting selections, order and decoding settings; original prompt replaces bare prompt.')
    plan=deepcopy(old)
    plan.update(protocol=PROTOCOL,prompt=original_prompt(name),collection_prompt=original_prompt(name),
                cohort_sha256=night.vanilla.fingerprint(new_manifest),parent_config_sha256=winners.digest(config),
                parent_cohort_sha256=old['cohort_sha256'])
    folder=output/'diffa'/name
    winners.freeze(folder/'cohort.json',new_manifest)
    winners.freeze(folder/'run_config.json',plan)
    plans={}
    for e in night.EMOTIONS:
        previous=json.loads((parent/e/'run_config.json').read_text())
        if previous['collection_ids']!=plan['fitting_ids'][e] or any(manifest['predictions'][q][e]!=e for q in previous['collection_ids']):
            raise ValueError('Parent fitting set is not the exact e2v-agreed selection.')
        p=deepcopy(previous)
        p.update(protocol=PROTOCOL,prompt=plan['prompt'],collection_prompt=plan['prompt'],cohort_sha256=plan['cohort_sha256'],
                 parent_config_sha256=winners.digest(parent/e/'run_config.json'))
        winners.freeze(folder/e/'run_config.json',p)
        plans[e]=p
    print(f'{name}: {len(plan["question_order"])} questions × neutral/angry/happy/sad; fresh block collection; L31 answer +1.',flush=True)
    print('Prompt:',plan['prompt'],flush=True)
    for e in night.EMOTIONS:
        print(f'  {e}: same {len(plans[e]["collection_ids"])} e2v-agreed fitting IDs; '
              f'{len(plan["question_order"])-len(plans[e]["collection_ids"])} unchanged holdout IDs.',flush=True)
    return folder,plan,plans,parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','run','summary'),default='run')
    parser.add_argument('--datasets',choices=('bbh3','bbh5'),nargs='+',default=['bbh3','bbh5'])
    parser.add_argument('--source',type=Path,default=Path('outputs/emotion-block-night'))
    parser.add_argument('--output',type=Path,default=Path('outputs/emotion-bbh-original-prompt'))
    parser.add_argument('--audio-root',type=Path)
    args=parser.parse_args()
    if args.mode=='summary':
        for name in args.datasets:
            folder=args.output/'diffa'/name
            if not (folder/'run_config.json').exists():
                print(f'{name}: original-prompt comparison not prepared yet.',flush=True)
                continue
            plan=json.loads((folder/'run_config.json').read_text())
            plans={e:json.loads((folder/e/'run_config.json').read_text()) for e in night.EMOTIONS}
            night.report(folder,plan,plans)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'runner.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cases=[prepare(name,args.source,args.output) for name in args.datasets]
        for folder,plan,_,_ in cases:
            winners.score_rows(dict(target=plan),[],folder/'scoring-preflight')
        if args.mode=='prepare':
            print('Prepared metadata only; no old vectors or responses reused.',flush=True)
            return
        registry=night.Registry(night.audio_root(args.audio_root))
        runtime_cases=[]
        for folder,plan,plans,parent in cases:
            # Resolve the exact same bytes even when using a staged Jenny directory.
            runtime=deepcopy(plan)
            runtime['records']=[registry.relocate(r) for r in plan['records']]
            previous_models=json.loads((parent/'runtime.json').read_text())['model_revisions']
            if model_revisions()!=previous_models:
                raise ValueError('Active model revision receipts differ from the bare-prompt run.')
            runtime_cases.append((folder,plan,plans,runtime))
        with Path('outputs/jenny-task-vectors.lock').open('a') as gpu:
            print('Waiting for shared DIFFA GPU lock and idle GPU.',flush=True)
            fcntl.flock(gpu,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                time.sleep(30)
            backend=night.DiffaBackend()
            for folder,plan,plans,runtime in runtime_cases:
                winners.atomic_text(folder/'runtime.json',json.dumps(dict(runner_sha256=winners.digest(__file__),
                    block_collector_sha256=winners.digest(night.blocks.__file__),model_revisions=model_revisions(),
                    packages={p:version(p) for p in ('torch','transformers')},
                    model_config=backend.model.config.to_dict()),indent=2,default=str)+'\n')
                night.generate_dataset(folder,runtime,plans,backend)
                night.natural_report(folder,plan)
                night.steer_dataset(folder,runtime,plans,backend)
                print(f'DONE {plan["name"]}: original-prompt all-emotion evaluation, new block vectors and x+1 steering.',flush=True)


if __name__=='__main__':
    main()
