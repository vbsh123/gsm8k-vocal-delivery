"""Inventory surviving experiment audio and classify full recordings with emotion2vec/SenseVoice."""
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib
from importlib.metadata import version,PackageNotFoundError
import json
import os
from pathlib import Path
import time

import pack_experiment_backup as backup
from steer import read_rows

CONFIGS={'plan.jsonl','run_config.json','split.json','collection-config.json','data_config.json'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def inventory(repo,output):
    repo=repo.resolve();output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    files=[p for p in backup.walk(repo/'outputs') if output not in p.parents]
    configs=[p for p in files if p.name in CONFIGS];index=backup.source_audio_index(configs)
    hashes={};clips={};issues=[]
    def hashed(path):
        path=path.resolve()
        if path not in hashes:hashes[path]=sha(path)
        return hashes[path]
    def add(path,alias):
        path=path.resolve();key=hashed(path)
        clip=clips.setdefault(key,dict(clip_id=key,audio_sha256=key,audio_file=str(path),paths=[],aliases=[]))
        if str(path) not in clip['paths']:clip['paths'].append(str(path))
        if alias not in clip['aliases']:clip['aliases'].append(alias)
    for i,config in enumerate(configs,1):
        try:
            data,_=backup.snapshot(config)
            plan=[json.loads(s) for s in data.splitlines() if s.strip()] if config.suffix=='.jsonl' else json.loads(data)
            rows=backup.candidates(plan)
        except (ValueError,OSError,TypeError) as exc:
            issues.append(dict(source=str(config),error=str(exc)));continue
        for row in rows:
            values=[row[k] for k in ('audio_file','audio_path') if isinstance(row.get(k),str) and Path(row[k]).suffix.lower() in backup.AUDIO]
            if not values:continue
            direct=[];indexed=[]
            for value in values:
                path=Path(value);direct.extend([path] if path.is_absolute() else [config.parent/path,repo/path])
                indexed.extend(index.get(backup.audio_key(row,path.name),[]))
            expected=row.get('audio_sha256');chosen=None
            try:
                for path in direct:
                    if path.is_file() and (not expected or hashed(path)==expected):chosen=path;break
                if chosen is None:
                    matches={p.resolve():p for p in indexed if p.is_file() and (not expected or hashed(p)==expected)}
                    if matches and (expected or len({hashed(p) for p in matches})==1):chosen=next(iter(matches))
                if chosen is None:raise FileNotFoundError('Missing/ambiguous audio or SHA256 mismatch; no prediction fabricated.')
                add(chosen,dict(source_config=str(config),question_id=row.get('question_id',row.get('clip_id')),
                    voice=row.get('voice',row.get('speaker','unknown')),requested_style=row.get('style','unknown'),
                    question=row.get('question'),answer=row.get('answer',row.get('reference_answer')),request=row.get('request'),
                    original_audio_reference=values[0],expected_audio_sha256=expected))
            except (OSError,ValueError) as exc:
                issues.append(dict(source=str(config),question_id=row.get('question_id'),audio_references=values,error=str(exc)))
        if i%100==0:print(f'Inventory {i}/{len(configs)} configs; {len(clips)} unique recordings.',flush=True)
    for path in files:
        if path.suffix.lower() not in backup.AUDIO:continue
        try:
            if not path.is_file():raise FileNotFoundError('Missing file/broken audio link.')
            real=path.resolve()
            if real in hashes and hashes[real] in clips:continue
            add(path,dict(source_config=None,question_id=None,voice='unknown',requested_style='unknown',
                original_audio_reference=str(path),expected_audio_sha256=None))
        except OSError as exc:issues.append(dict(source=str(path),error=str(exc)))
    legacy=[]
    for path in files:
        if path.suffix.lower() not in {'.jsonl','.json','.csv','.tsv','.txt'}:continue
        if not any(word in str(path.relative_to(repo/'outputs')).lower() for word in ('emotion2vec','sensevoice','dataset-emotions')):continue
        try:legacy.append(dict(path=str(path),sha256=sha(path)))
        except OSError as exc:issues.append(dict(source=str(path),error=str(exc)))
    records=sorted(clips.values(),key=lambda r:r['clip_id'])
    tmp=output/'audio_inventory.jsonl.tmp';tmp.write_text(''.join(json.dumps(r)+'\n' for r in records));tmp.replace(output/'audio_inventory.jsonl')
    manifest=dict(protocol='available-audio-full-recording-audit-v1',repo=str(repo),unique_audio=len(records),
        aliases=sum(len(r['aliases']) for r in records),source_configs=len(configs),issues=issues,legacy_predictions=legacy,
        inventory_sha256=sha(output/'audio_inventory.jsonl'),created_unix=time.time(),
        policy='All surviving outputs audio and resolvable experiment references, no outcome/emotion filtering. '
            'Deduplicate byte-identical audio by SHA256; retain aliases. Missing original audio cannot be reclassified. '
            'Classify whole mono16kHz recordings; no cropping/chunk fallback. Legacy labels archived separately, not silently reused.')
    write_json(output/'inventory_manifest.json',manifest)
    print(f'Inventory: {len(records)} unique recordings; {len(issues)} missing/unreadable/mismatched references. Legacy label files: {len(legacy)}.',flush=True)
    return records


def model_receipt(adapter):
    hub=Path(os.environ.get('HF_HUB_CACHE',str(Path(os.environ.get('HF_HOME',Path.home()/'.cache/huggingface'))/'hub')))
    ref=hub/('models--'+adapter.MODEL.replace('/','--'))/'refs/main'
    return dict(model=adapter.MODEL,adapter_sha256=sha(adapter.__file__),cached_hf_main_revision=ref.read_text().strip() if ref.is_file() else None)


def classify(output,model,device,retry_errors=False):
    import torch
    import soundfile as sf
    import librosa
    torch.set_num_threads(4)
    clips=read_rows(output/'audio_inventory.jsonl');path=output/f'results-{model}.jsonl'
    adapter=importlib.import_module(model+'_agreement');saved={}
    for row in read_rows(path,repair=True):
        if row['model']!=adapter.MODEL or row['processing_mode']!='whole_recording':raise ValueError('Saved classifier/mode differs.')
        saved[row['clip_id']]=row
    pending=[r for r in clips if r['clip_id'] not in saved or retry_errors and saved[r['clip_id']]['classification_status']!='classified']
    print(f'{model}: {len(clips)} unique clips; {len(pending)} pending; device={device}.',flush=True)
    inventory_hash=sha(output/'audio_inventory.jsonl')
    predictor=adapter.load_predictor(device) if pending else None
    packages={}
    for name in ('torch','torchaudio','funasr','librosa','soundfile','huggingface-hub'):
        try:packages[name]=version(name)
        except PackageNotFoundError:pass
    write_json(output/f'runtime-{model}.json',dict(**model_receipt(adapter),device=device,packages=packages,
        inventory_sha256=inventory_hash,script_sha256=sha(__file__),
        note='Whole recordings; float32 mono16kHz; no classifier/outcome filtering. Errors recorded explicitly; no chunk fallback.'))
    with path.open('a') as stream:
        for i,clip in enumerate(pending,1):
            base=dict(clip_id=clip['clip_id'],audio_sha256=clip['audio_sha256'],model=adapter.MODEL,
                processing_mode='whole_recording',inventory_sha256=inventory_hash)
            try:
                audio_file=next((Path(p) for p in clip['paths'] if Path(p).is_file()),Path(clip['audio_file']))
                if sha(audio_file)!=clip['audio_sha256']:raise ValueError('Audio changed after inventory.')
                audio,rate=sf.read(audio_file,dtype='float32')
                if audio.ndim==2:audio=audio.mean(axis=1)
                if rate!=16000:audio=librosa.resample(audio,orig_sr=rate,target_sr=16000)
                base.update(audio_file=str(audio_file),sample_rate=16000,samples=len(audio),duration_seconds=len(audio)/16000)
                if len(audio)<400:raise ValueError('Insufficient audio samples (<400); no padding or invented label.')
                prediction=predictor(audio)
                if device.startswith('cuda'):torch.cuda.synchronize()
                row=dict(base,classification_status='classified',**prediction)
            except Exception as exc:
                row=dict(base,classification_status='error',error=f'{type(exc).__name__}: {exc}')
                if device.startswith('cuda'):torch.cuda.empty_cache()
            stream.write(json.dumps(row)+'\n');stream.flush()
            print(f'{model} [{i}/{len(pending)}] {clip["audio_file"]}: {row.get("emotion",row.get("error"))}',flush=True)
    summarize(output)


def summarize(output):
    clips=read_rows(output/'audio_inventory.jsonl');all_summary={}
    for model in ('emotion2vec','sensevoice'):
        saved={r['clip_id']:r for r in read_rows(output/f'results-{model}.jsonl')};groups=defaultdict(Counter)
        classified=sum(saved.get(c['clip_id'],{}).get('classification_status')=='classified' for c in clips)
        errors=sum(saved.get(c['clip_id'],{}).get('classification_status')=='error' for c in clips)
        print(f'\n{model}: {classified}/{len(clips)} classified; {errors} errors; {len(clips)-classified-errors} pending.')
        for clip in clips:
            prediction=saved.get(clip['clip_id'],{})
            if prediction.get('classification_status')!='classified':continue
            for voice,style in {(a.get('voice','unknown'),a.get('requested_style','unknown')) for a in clip['aliases']}:
                groups[str(voice),str(style)][prediction['emotion']]+=1
        rows=[]
        for (voice,style),counts in sorted(groups.items()):
            n=sum(counts.values());match=counts[style]/n if style in ('neutral','angry','happy','sad') else None
            rows.append(dict(voice=voice,requested_style=style,n=n,counts=dict(counts),match_rate=match))
            print(voice,style,f'N={n}',dict(counts),'match='+('n/a' if match is None else f'{match:.1%}'))
        all_summary[model]=dict(total_unique_audio=len(clips),classified=classified,errors=errors,groups=rows)
    write_json(output/'summary.json',dict(models=all_summary,note='Unique audio hash per voice/requested style. Classifier labels are diagnostics, not human ground truth. No response filtering.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('inventory','classify','summary'),required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/available-audio-emotions'))
    p.add_argument('--model',choices=('emotion2vec','sensevoice'),default='emotion2vec')
    p.add_argument('--device',default='cuda:0');p.add_argument('--retry-errors',action='store_true');args=p.parse_args()
    if args.mode=='summary':summarize(args.output);return
    import fcntl
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'audit.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='inventory':inventory(Path.cwd(),args.output)
        else:classify(args.output,args.model,args.device,args.retry_errors)


if __name__=='__main__':main()
