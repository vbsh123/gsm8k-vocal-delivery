"""Inventory remote output artifacts without loading models or reading tensor caches."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path


def inventory(root):
    runs=[]
    configs=sorted({p for pattern in ('run_config.json','split.json','data_config.json') for p in root.rglob(pattern)})
    for config in configs:
        folder=config.parent
        try:plan=json.loads(config.read_text())
        except (OSError,ValueError):plan={}
        files=[]
        for p in sorted(folder.iterdir()):
            if not p.is_file() or p.suffix not in ('.json','.jsonl','.csv','.pt'):continue
            item=dict(path=str(p),bytes=p.stat().st_size)
            if p.suffix=='.jsonl':
                with p.open('rb') as stream:item['nonempty_lines']=sum(bool(line.strip()) for line in stream)
            files.append(item)
        runs.append(dict(folder=str(folder),config=str(config),protocol=plan.get('protocol'),task=plan.get('task',plan.get('dataset')),
            layer=plan.get('layer'),strength=plan.get('strength',plan.get('strengths')),artifacts=files,
            cached_tensor_files=sum(1 for p in folder.rglob('*.pt'))))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),root=str(root.resolve()),runs=runs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs'))
    parser.add_argument('--output',type=Path,default=Path('/workspace/experiment-output-inventory.json'))
    args=parser.parse_args();result=inventory(args.root);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'{len(result["runs"])} configs inventoried. Download: {args.output.resolve()}')


if __name__=='__main__':main()
