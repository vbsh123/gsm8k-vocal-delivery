"""Package Phi neutral/angry evidence only; exclude steering, vectors and diagnostics."""
from collections import Counter
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path

import pack_experiment_backup as backup

MODEL='microsoft/Phi-4-multimodal-instruct'
PATTERNS=(
    'gsm8k-candidates/phi4/*/run_config.json',
    'bbh-three-candidates/phi4/*/run_config.json',
    'bbh-five-candidates/phi4/*/run_config.json',
    'phi4-sweeps/*/*/run_config.json',
    'phi4-task-steering/*/*/run_config.json',
    'phi4-gsm400-all-layers-test869-minus1/run_config.json',
)


def sha(data):return hashlib.sha256(data).hexdigest()
def identity(row):return row['question_id'],row['voice'],row.get('condition',row.get('style'))


def candidate(config,outputs):
    content,_=backup.snapshot(config);plan=json.loads(content)
    if plan.get('recipe',{}).get('model')!=MODEL:return None
    if any(word in str(config.relative_to(outputs)) for word in ('answer-only','reasoning-emotion','instruction-controls')):return None
    expected={identity(r):r for r in plan.get('recordings',[]) if identity(r)[2] in ('neutral','angry')}
    if {k[2] for k in expected}!={'neutral','angry'}:return None
    task=plan.get('task','gsm8k' if config.parent.name=='phi4-gsm400-all-layers-test869-minus1' else None)
    if task is None:raise ValueError('Unknown Phi delivery task: '+str(config))
    references=config.parent.name=='phi4-gsm400-all-layers-test869-minus1'
    paths=sorted(config.parent.glob('references*.jsonl')) if references else [config.parent/'results.jsonl']
    done={};receipts=[]
    for path in paths:
        if not path.is_file():continue
        raw,trimmed=backup.snapshot(path);receipts.append(dict(path=str(path),sha256=sha(raw),incomplete_tail_omitted=trimmed))
        for number,line in enumerate(raw.splitlines(),1):
            if not line.strip():continue
            row=json.loads(line);key=identity(row)
            if key[2] not in ('neutral','angry'):continue
            if row.get('interventions') or float(row.get('strength') or 0)!=0:
                raise ValueError('An allegedly unsteered row has an intervention: '+str(path))
            if key not in expected:raise ValueError('Baseline outside frozen source cohort: '+str(path))
            for field,value in (('prompt',plan['prompt']),('seed',plan['seed']),('max_new_tokens',plan['max_new_tokens']),
                                ('model',MODEL),('revision',plan['recipe']['revision']),('audio_sha256',expected[key]['audio_sha256'])):
                if row.get(field)!=value:raise ValueError('Baseline recipe/audio mismatch: '+field+' in '+str(path))
            if key in done:
                previous=json.loads(done[key]['raw'])
                if any(previous.get(k)!=row.get(k) for k in ('response','response_tokens','reference_answer')):
                    raise ValueError('Conflicting baseline responses within source run: '+str(path))
                continue
            done[key]=dict(raw=line.decode(),source=str(path),line=number,sha256=sha(line))
    if not done:return None
    signature=dict(task=task,prompt=plan['prompt'],recipe=plan['recipe'],seed=plan['seed'],max_new_tokens=plan['max_new_tokens'],
        expected=[dict(question_id=k[0],voice=k[1],style=k[2],audio_sha256=r['audio_sha256']) for k,r in sorted(expected.items())])
    group=sha(json.dumps(signature,sort_keys=True).encode())
    styles={s:{k[:2] for k in done if k[2]==s} for s in ('neutral','angry')}
    return dict(group=group,task=task,plan=plan,expected=expected,rows=done,source_config=str(config),
        source_config_sha256=sha(content),source_files=receipts,matched=len(styles['neutral']&styles['angry']))


def export(repo):
    outputs=repo/'outputs';choices={}
    for config in sorted({p for pattern in PATTERNS for p in outputs.glob(pattern)}):
        run=candidate(config,outputs)
        if run is None:continue
        # Select the most complete copy of an identical frozen recipe/cohort;
        # never select by correctness, response length or any metric.
        rank=(run['matched'],len(run['rows']))
        old=choices.get(run['group'])
        if old is None or rank>(old['matched'],len(old['rows'])):choices[run['group']]=run
    if not choices:raise FileNotFoundError('No saved Phi neutral/angry responses found in known baseline sources.')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target=outputs/'phi-deliveries-only'/stamp;target.mkdir(parents=True)
    inventory=[]
    for group,run in sorted(choices.items()):
        plan=run['plan'];folder=target/run['task']/group[:16];folder.mkdir(parents=True)
        expected=run['expected'];keys=sorted(run['rows']);qids={k[0] for k in expected}
        clean=dict(protocol='phi-neutral-angry-evidence-export-v1',task=run['task'],prompt=plan['prompt'],
            prompt_variant=plan.get('prompt_variant','saved-task-prompt'),recipe=plan['recipe'],seed=plan['seed'],
            max_new_tokens=plan['max_new_tokens'],bank={q:plan['bank'][q] for q in qids},
            recordings=[expected[k] for k in sorted(expected)],interventions=[],
            selection='Baseline-only derivative export. Original frozen cohort/recipe retained; no outcome filtering. Source selection follows.',
            source_selection=plan.get('selection'),source_config=run['source_config'],source_config_sha256=run['source_config_sha256'],
            note='Raw lines retain original metadata/config hashes. Only neutral/angry rows with zero strength and no interventions. '
                'No source steering configuration, vector, activation, or steered response payload included. Distinct overlapping cohorts remain separate.')
        (folder/'run_config.json').write_text(json.dumps(clean,indent=2)+'\n')
        (folder/'results.jsonl').write_text(''.join(run['rows'][k]['raw']+'\n' for k in keys))
        provenance=dict(source_config=run['source_config'],source_config_sha256=run['source_config_sha256'],source_files=run['source_files'],
            exported_lines=[{k:v for k,v in run['rows'][key].items() if k!='raw'} for key in keys])
        (folder/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
        counts=dict(Counter(k[2] for k in keys));item=dict(task=run['task'],folder=str(folder.relative_to(target)),
            source_config=run['source_config'],prompt=plan['prompt'],max_new_tokens=plan['max_new_tokens'],
            expected_questions=len(qids),saved=counts,matched=run['matched'])
        inventory.append(item);print(run['task'],counts,'matched',run['matched'],'/',len(qids),flush=True)
    (target/'INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    (target/'README.md').write_text('Phi neutral/angry delivery evidence only. Each subfolder retains one frozen cohort and recipe.\n'
        'Inspect INVENTORY.json for prompts, completion and overlap; do not add matched counts across overlapping cohorts.\n'
        'Rows are copied verbatim with source-file and line receipts. Missing/unpaired answers remain visible.\n'
        'No steering vectors, activation caches, layer rankings, steered answers or reasoning-control ablations are exported.\n')
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=Path('/workspace'))
    parser.add_argument('--samples',type=int,default=3)
    parser.add_argument('--sample-mb',type=int,default=100)
    args=parser.parse_args();repo=Path.cwd().resolve();target=export(repo)
    backup.GROUPS['phi-deliveries']=(str(target.relative_to(repo/'outputs')),)
    backup.package(repo,args.destination,samples=args.samples,sample_mb=args.sample_mb,only=['phi-deliveries'])
