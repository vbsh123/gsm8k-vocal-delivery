"""Evaluate the fixed original L12 acoustic x1 direction on new Azure neutral audio."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from steer import read_rows


def select_recordings(folder, count, sf):
    selected={}
    for row in sorted(read_rows(folder/'plan.jsonl'),key=lambda r:r['question_id']):
        if row['provider']!='azure' or row['style']!='neutral':
            continue
        path=(folder/row['audio_path']).resolve()
        seconds=sf.info(path).duration
        if not 0<seconds<=30:
            continue
        selected.setdefault(row['question_id'],dict(
            clip_id=f"{row['question_id']}__{row['voice']}__neutral",question_id=row['question_id'],
            audio_path=str(path),answer=str(row['answer']).rsplit('####',1)[-1].strip(),
            style='neutral',speaker=row['voice'],repeat=0,duration_seconds=seconds))
    records=list(selected.values())[:count] if count else list(selected.values())
    if count<0 or not records or (count and len(records)!=count):
        raise ValueError(f'Need {count or "all"} eligible neutral questions; found {len(records)}')
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--path',type=Path,default=Path('outputs/azure-jenny-remaining-test'))
    p.add_argument('--num-examples',type=int,default=200)
    p.add_argument('--vectors',type=Path)
    p.add_argument('--output',type=Path,default=Path('outputs/steer-jenny-remaining200-L12'))
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    if args.summary_only:
        subprocess.run([sys.executable,'steer_audio.py','--output',str(args.output),'--summary-only'],check=True)
        return
    import soundfile as sf
    vector=args.vectors
    if vector is None:
        previous=json.loads(Path('outputs/steer-jenny-angry-complete/run_config.json').read_text())
        vector=Path(previous['settings']['vectors'])
    if not vector.is_file():
        p.error('Original vector missing at its saved path. Pass --vectors /actual/path/style-vectors.pt')
    records=select_recordings(args.path,args.num_examples,sf)
    conditions=[dict(name='baseline',interventions=[]),dict(name='L12-acoustic-a1',
        interventions=[dict(layer=12,group='acoustic',strength=1)])]
    args.output.mkdir(parents=True,exist_ok=True)
    plan=args.output/'selected-plan.json'
    value=dict(recordings=records,conditions=conditions,
               selection='First eligible new Azure neutral questions; no correctness or classifier filtering')
    if plan.exists() and json.loads(plan.read_text())!=value:
        p.error('Frozen question selection changed; use another output folder')
    plan.write_text(json.dumps(value,indent=2)+'\n')
    command=[sys.executable,'-u','steer_audio.py','--plan',str(plan),'--vectors',str(vector),
             '--output',str(args.output)]
    if args.prepare_only:
        command.append('--prepare-only')
    print(f'{len(records)} new neutral recordings; original L12 acoustic x1; baseline and steering; seed42.',flush=True)
    subprocess.run(command,check=True)


if __name__=='__main__':
    main()
