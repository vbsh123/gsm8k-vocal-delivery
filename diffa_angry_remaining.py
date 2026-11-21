"""Sequential angry-only agreed halves on SVAMP, MultiArith and IFEval."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import diffa_angry_agreed as audit
import diffa_bbh_angry125 as half
import diffa_happy_nightly as collection
from steer import read_rows

DATASETS=('svamp','multiarith','ifeval')
HALVES={'svamp':500,'multiarith':290,'ifeval':254}
OUTPUT=Path('outputs/diffa-angry-only-remaining')


def export_svamp(destination):
    """Run on Phi: package existing test-half angry audio, never regenerate it."""
    config=Path('outputs/phi4-next-tasks/svamp/data_config.json')
    data=json.loads(config.read_text());source=Path(data['source'])
    plans={r['question_id']:r for r in read_rows(source/'plan.jsonl') if r.get('style')=='angry'}
    rows=[];receipts=[];files=[]
    if len(data['test'])!=500:raise ValueError('Expected existing Phi SVAMP test half of500.')
    for pair in data['test']:
        q=pair['question_id'];record=pair['angry'];path=Path(record['audio_file']);row=plans[q]
        if audit.winners.digest(path)!=record['audio_sha256']:raise ValueError('Source angry audio changed.')
        if row.get('voice')!='en-US-JennyNeural' or row.get('provider')!='azure':raise ValueError('Expected original Jenny Azure source.')
        relative=f'audio/{q}-angry.wav'
        rows.append(dict(row,audio_path=relative));files.append((path,relative))
        receipts.append(dict(question_id=q,audio_path=relative,sha256=record['audio_sha256']))
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path,relative in files:z.write(path,relative)
        z.writestr('plan.jsonl',''.join(json.dumps(r)+'\n' for r in rows))
        z.writestr('receipts.json',json.dumps(dict(source_config_sha256=audit.winners.digest(config),recordings=receipts),indent=2)+'\n')
    temporary.replace(destination)
    print(f'Exported500 existing angry clips: {destination}; no inference or TTS.',flush=True)


def audit_and_prepare(args):
    root=args.output;aroot=root/'audit'
    if args.mode=='audit':
        # Isolate this audit from the already-running BBH experiment's audit snapshot.
        command=[sys.executable,'-u','diffa_angry_agreed.py','--mode','audit','--output',str(root),
                 '--device',args.device,'--datasets',*args.datasets]
        subprocess.run(command,check=True)
        return None
    data=json.loads((aroot/'cohorts.json').read_text())['datasets'];labels=audit.predictions(aroot);plans={}
    for name in args.datasets:
        folder=root/name;path=folder/'run_config.json'
        if path.exists():plan=json.loads(path.read_text())
        else:plan=half.prepare(name,data[name],labels,aroot,folder,half=HALVES[name])
        old=plan['original_target']
        if (audit.winners.digest(old['config_file'])!=old['config_sha256'] or
                audit.winners.digest(old['vector'])!=old['vector_sha256']):raise ValueError('Original source receipts changed.')
        audit.winners.score_rows(dict(target=plan['target']),[],folder)
        plans[name]=plan
    return plans


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('audit','prepare','run','summary','export-svamp'),default='run')
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--export',type=Path,default=Path('/workspace/svamp-phi-angry-test500.zip'))
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if args.mode=='export-svamp':export_svamp(args.export);return
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='summary':
        for name in args.datasets:
            folder=args.output/name;path=folder/'run_config.json'
            if not path.exists():print(name,'not prepared.');continue
            if (folder/'primary/run_config.json').exists():
                collection.report_steering(folder/'primary');half.compare_original(json.loads(path.read_text()),folder/'primary')
            else:print(name,'evaluation not started.')
        return
    with (args.output/'remaining.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plans=audit_and_prepare(args)
        if args.mode in ('audit','prepare'):return
        if shutil.disk_usage(args.output).free<600*2**20:raise RuntimeError('Need600 MiB free for aggregate checkpoints.')
        with Path('outputs/jenny-task-vectors.lock').open('a') as lock:
            print('Waiting for existing GPU jobs, then '+ ' -> '.join(args.datasets)+' sequentially.',flush=True)
            fcntl.flock(lock,fcntl.LOCK_EX)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(30)
            from extract_vectors import load_model
            model,tokenizer,features=load_model()
            audit.winners.atomic_text(args.output/'runtime.json',json.dumps(dict(runner_sha256=audit.winners.digest(__file__),
                collector_sha256=audit.winners.digest(collection.__file__),datasets=args.datasets,model_config=model.config.to_dict()),indent=2,default=str)+'\n')
            for name,plan in plans.items():
                folder=args.output/name;collection.collect(plan,folder,model,tokenizer,features)
                source=audit.descriptor(plan,folder);job=audit.job_for(source,plan['target'],[0.,1.])
                destination=folder/'primary';destination.mkdir(exist_ok=True)
                audit.winners.freeze(destination/'run_config.json',job);audit.winners.reuse_answers(job,destination)
                half.fill_neutral(plan,job,destination,model,tokenizer,features)
                audit.winners.generate_job(job,destination,model,tokenizer,features)
                collection.report_steering(destination,tokenizer);half.compare_original(plan,destination)


if __name__=='__main__':main()
