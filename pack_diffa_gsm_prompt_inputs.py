"""Small supplemental original-GSM vector/result bundle for the new DIFFA prompt ablation."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile

from pack_diffa_migration import snapshot

MANIFEST='DIFFA_GSM_PROMPT_INPUTS_MANIFEST.json'


def inputs(repo):
    import diffa_happy_nightly as task
    source=repo/'outputs/diffa-happy-winners/gsm8k/run_config.json'
    plan=json.loads(source.read_text())
    if len(plan['collection_ids'])!=425 or len(plan['test_ids'])!=718 or plan['emotion']!='happy':
        raise ValueError('Need the completed ORIGINAL GSM happy425/718 plan.')
    angry=Path(plan['target']['vector'])
    happy=source.parent/'style-vectors.pt'
    for path,emotion in ((angry,'angry'),(happy,'happy')):
        _,info=task.winners.read_vector(path,emotion)
        if info['pairs']!=425:
            raise ValueError(f'Expected original425-question vector: {path}')
    files={source,angry,happy,Path(plan['target']['config_file'])}
    for folder in (source.parent/'strength-1',source.parent/'delivery',
                   repo/'outputs/winning-strengths/gsm8k',Path(plan['target']['run'])):
        for name in ('run_config.json','split.json','results.jsonl'):
            if (folder/name).is_file():files.add(folder/name)
    return {p.resolve():snapshot(p) for p in files}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=Path('/workspace'))
    args=parser.parse_args()
    repo=Path.cwd().resolve()
    files=inputs(repo)
    args.destination.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path=args.destination/f'diffa-gsm-original-prompt-inputs-{stamp}.zip'
    manifest=dict(repo=str(repo),scope='Original GSM angry/happy vectors, frozen cohort metadata and saved answers only; no audio or model weights.',files=[])
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for source,data in sorted(files.items()):
            member=source.relative_to(repo).as_posix()
            archive.writestr(member,data)
            manifest['files'].append(dict(path=member,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        archive.writestr(MANIFEST,json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(path) as archive:
        if archive.testzip():raise ValueError('ZIP integrity check failed.')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.zip.sha256').write_text(f'{digest}  {path.name}\n')
    print(f'DOWNLOAD: {path}\nSIZE: {path.stat().st_size/2**20:.2f} MiB\nCHECKSUM: {path}.sha256')
    print('Upload ZIP and checksum to /workspace on the NEW DIFFA machine, then restore_diffa_bundle.py this ZIP.')


if __name__=='__main__':main()
