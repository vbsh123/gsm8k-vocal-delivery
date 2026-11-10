"""Audit existing dataset audio with emotion2vec or SenseVoice; no answer generation.

One result per unique audio file, retaining its question IDs and source category.
Long recordings use every 30-second chunk: duration-weighted scores (emotion2vec)
or duration-weighted label votes (SenseVoice). Audio is never cropped to 30s.
"""

import argparse
from collections import Counter, defaultdict
import importlib
import io
import json
from pathlib import Path
import tarfile
import zipfile


REPOS={
    'spoken-gsm':['mistralai/gsm8k_speech'],
    'mmau':['gamma-lab-umd/MMAU-test','gamma-lab-umd/MMAU-test-mini'],
    'mmau-pro':['gamma-lab-umd/MMAU-Pro'],
    'speechr':[],
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def parquet_rows(path):
    import pyarrow.parquet as pq
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=1):
            yield from batch.to_pylist()


def audio_index(rows, dataset):
    """Collect repeated references without weighting repeated questions twice."""
    index={}
    for row in rows:
        paths=row['audio_path'] if dataset=='mmau-pro' else [row['audio_id']]
        category=row['category'] if dataset=='mmau-pro' else row['task']
        for name in paths:
            name=name.removeprefix('./')
            record=index.setdefault(name,dict(clip_id=name,audio_path=name,question_ids=[],categories=[]))
            if row['id'] not in record['question_ids']:
                record['question_ids'].append(row['id'])
            if category not in record['categories']:
                record['categories'].append(category)
    return index


def archived_recordings(archive, index, completed):
    """Read an archive once, without extracting another copy of the audio."""
    found=set()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                name=member.filename.removeprefix('./')
                if name in index:
                    found.add(name)
                    if name not in completed:
                        yield index[name],zipped.read(member)
    else:
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                name=member.name.removeprefix('./')
                if member.isfile() and name in index:
                    found.add(name)
                    if name not in completed:
                        yield index[name],tar.extractfile(member).read()
    missing=set(index)-found
    if missing:
        raise ValueError(f'{len(missing)} metadata audio paths missing from archive; examples: {sorted(missing)[:3]}')


def recordings(args, source, completed):
    if args.dataset=='speechr':
        # This includes all locally downloaded delivery variants, each once.
        found=0
        for path in sorted(args.audio_root.rglob('*')):
            if '__MACOSX' in path.parts or path.name.startswith('._') or not path.is_file() or path.suffix.lower() not in ('.wav','.flac','.mp3','.ogg'):
                continue
            found+=1
            name=path.relative_to(args.audio_root).as_posix()
            if name not in completed:
                yield dict(clip_id=name,audio_path=name,question_ids=[path.stem],categories=[path.parent.name]),path
        if not found:
            raise ValueError('No SpeechR audio files found. Unzip the nested audio_core archives first.')
        return
    from huggingface_hub import hf_hub_download
    def download(repo,name):
        return Path(hf_hub_download(repo,name,repo_type='dataset',revision=source['revisions'][repo]))
    if args.dataset=='spoken-gsm':
        repo=REPOS[args.dataset][0]
        index=0
        for name in source['files'][repo]:
            for row in parquet_rows(download(repo,name)):
                cid=f'spoken_gsm_test_{index:05d}'
                index+=1
                if cid not in completed:
                    yield dict(clip_id=cid,question_ids=[cid],categories=['speech']),row['audio']['bytes']
        return
    for repo in REPOS[args.dataset]:
        if repo.endswith('test-mini'):
            for i,row in enumerate(parquet_rows(download(repo,'test_mini.parquet'))):
                attrs=json.loads(row['other_attributes'])
                cid='test-mini/'+attrs['id']
                if cid not in completed:
                    yield dict(clip_id=cid,question_ids=[attrs['id']],categories=[attrs['task']]),row['context']['bytes']
            continue
        rows=list(parquet_rows(download(repo,'test.parquet')))
        index=audio_index(rows,args.dataset)
        print(f'{repo}: {len(index)} unique audio paths referenced by {len(rows)} questions.',flush=True)
        if args.audio_root:
            for cid,record in index.items():
                if cid not in completed:
                    path=args.audio_root/record['audio_path']
                    if not path.exists():
                        raise FileNotFoundError(f'Expected metadata path {path}; point --audio-root at the parent of data/ or test-audios/.')
                    yield record,path
        else:
            archive=download(repo,'data.zip' if args.dataset=='mmau-pro' else 'test-audios.tar.gz')
            yield from archived_recordings(archive,index,completed)


def aggregate_predictions(predictions, durations, model):
    if model=='emotion2vec':
        labels=predictions[0]['labels']
        if any(p['labels']!=labels for p in predictions):
            raise ValueError('Classifier label order changed between chunks')
        scores=[sum(p['scores'][i]*d for p,d in zip(predictions,durations))/sum(durations)
                for i in range(len(labels))]
        return importlib.import_module('emotion2vec_agreement').parse_prediction(dict(labels=labels,scores=scores))
    votes=Counter()
    for prediction,duration in zip(predictions,durations):
        votes[prediction['emotion']]+=duration
    return dict(emotion=votes.most_common(1)[0][0],duration_votes=dict(votes))


def report(output, model):
    rows={r['clip_id']:r for r in read_rows(output/f'results-{model}.jsonl')}
    groups=defaultdict(Counter)
    for row in rows.values():
        groups['ALL'][row['emotion']]+=1
        for category in set(row['categories']):
            groups[category][row['emotion']]+=1
    print(f'\n{model}: {len(rows)} unique recordings classified; counts are per audio file, not question.')
    print(f'{"Category":24} {"Label":16} {"Clips":>8} {"Percent":>9}')
    for category,counts in groups.items():
        for label,count in counts.most_common():
            print(f'{category:24} {label:16} {count:8} {count/sum(counts.values()):9.1%}')
    print('Music/sound categories are classifier outputs, not validated speaker emotions. No accuracy evaluation was performed.')
    summary=dict(unique_recordings=len(rows),categories={k:dict(v) for k,v in groups.items()})
    other='sensevoice' if model=='emotion2vec' else 'emotion2vec'
    other_rows={r['clip_id']:r for r in read_rows(output/f'results-{other}.jsonl')}
    shared=rows.keys() & other_rows.keys()
    if shared:
        def label(value):
            return 'unknown' if value in ('unknown','<unk>','emo_unknown') else value
        agreed=Counter(label(rows[k]['emotion']) for k in shared
                       if label(rows[k]['emotion'])==label(other_rows[k]['emotion']))
        print(f'Both classifiers: {sum(agreed.values())}/{len(shared)} shared recordings agree ({sum(agreed.values())/len(shared):.1%}).')
        print('Agreed labels:',dict(agreed))
        summary['classifier_agreement']=dict(shared=len(shared),agreed=sum(agreed.values()),labels=dict(agreed))
    (output/f'summary-{model}.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=REPOS)
    p.add_argument('--model',choices=['emotion2vec','sensevoice'],default='emotion2vec')
    p.add_argument('--audio-root',type=Path,help='Existing unpacked dataset root. SpeechR requires this; HF audio otherwise downloads automatically.')
    p.add_argument('--output',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--limit',type=int,help='First N unique recordings, including completed results; omit for all.')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    output=args.output or Path('outputs/dataset-emotions')/args.dataset
    output.mkdir(parents=True,exist_ok=True)
    if args.summary_only:
        report(output,args.model)
        return
    if args.limit is not None and args.limit<1:
        p.error('--limit must be positive')
    if args.dataset=='speechr' and (args.audio_root is None or not args.audio_root.is_dir()):
        p.error('For SpeechR provide --audio-root outputs/speechr, containing the extracted WAVs.')
    from huggingface_hub import HfApi
    source_path=output/'source.json'
    root=str(args.audio_root.resolve()) if args.audio_root else None
    if source_path.exists():
        source=json.loads(source_path.read_text())
        if source['dataset']!=args.dataset or source['audio_root']!=root:
            p.error('Source changed; use a new output folder')
    else:
        api=HfApi()
        revisions={repo:api.dataset_info(repo).sha for repo in REPOS[args.dataset]}
        files={repo:sorted(name for name in api.list_repo_files(repo,repo_type='dataset',revision=revision)
                           if name.startswith('data/test-') and name.endswith('.parquet')) for repo,revision in revisions.items()}
        source=dict(dataset=args.dataset,audio_root=root,revisions=revisions,files=files)
        source_path.write_text(json.dumps(source,indent=2)+'\n')
    adapter=importlib.import_module(f'{args.model}_agreement')
    settings=dict(model=adapter.MODEL,sample_rate=16000,chunk_seconds=30,
                  aggregation='duration-weighted scores' if args.model=='emotion2vec' else 'duration-weighted votes')
    settings_path=output/f'settings-{args.model}.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:
        p.error('Classifier settings changed; use a new output folder')
    settings_path.write_text(json.dumps(settings,indent=2)+'\n')
    result_path=output/f'results-{args.model}.jsonl'
    saved={r['clip_id']:r for r in read_rows(result_path)}
    if args.limit is not None and len(saved)>=args.limit:
        report(output,args.model)
        return
    import librosa
    import soundfile as sf
    predictor=None
    completed=set(saved)
    with result_path.open('a') as stream:
        for record,payload in recordings(args,source,completed):
            if predictor is None:
                predictor=adapter.load_predictor(args.device)
            audio,sr=sf.read(io.BytesIO(payload) if isinstance(payload,bytes) else payload,dtype='float32')
            if audio.ndim==2:
                audio=audio.mean(axis=1)
            if sr!=16000:
                audio=librosa.resample(audio,orig_sr=sr,target_sr=16000)
            if not len(audio):
                raise ValueError(f'Empty audio: {record["clip_id"]}')
            predictions,durations=[],[]
            for start in range(0,len(audio),30*16000):
                chunk=audio[start:start+30*16000]
                predictions.append(predictor(chunk))
                durations.append(len(chunk)/16000)
            result=dict(**record,model=adapter.MODEL,duration_seconds=len(audio)/16000,
                        chunks=[dict(duration_seconds=d,**pred) for pred,d in zip(predictions,durations)],
                        **aggregate_predictions(predictions,durations,args.model))
            stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            stream.flush()
            completed.add(record['clip_id'])
            print(f'[{len(completed)}] {record["clip_id"]}: {result["emotion"]} ({len(predictions)} chunks)',flush=True)
            if args.limit is not None and len(completed)>=args.limit:
                break
    report(output,args.model)


if __name__=='__main__':
    main()
