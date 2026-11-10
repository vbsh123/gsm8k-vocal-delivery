"""Make five original Spoken GSM clips at 0.5x–2x for listening; CPU only."""

import argparse
from html import escape
import io
import json
from pathlib import Path
import zipfile

import librosa
import soundfile as sf


def original_clips(source):
    manifest=source/'manifest.jsonl'
    if manifest.exists():
        print('Using local original recordings.',flush=True)
        records={r['clip_id']:r for line in manifest.read_text().splitlines() if line.strip()
                 for r in [json.loads(line)] if r['style']=='original'}
        for record in records.values():
            path=source/record['audio_path']
            info=sf.info(path)
            if 0<info.frames/info.samplerate<=15:
                audio,rate=sf.read(path,dtype='float32')
                yield record['question_id'],audio,rate
        return

    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    dataset='mistralai/gsm8k_speech'
    print('No local originals found. Downloading public Spoken GSM data, not model weights.',flush=True)
    api=HfApi()
    revision=api.dataset_info(dataset).sha
    files=sorted(name for name in api.list_repo_files(dataset,repo_type='dataset',revision=revision)
                 if name.startswith('data/test-') and name.endswith('.parquet'))
    index=0
    for name in files:
        path=hf_hub_download(dataset,name,repo_type='dataset',revision=revision)
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=2,use_threads=False):
                for row in batch.to_pylist():
                    qid=f'spoken_gsm_test_{index:05d}'
                    index+=1
                    payload=io.BytesIO(row['audio']['bytes'])
                    info=sf.info(payload)
                    if 0<info.frames/info.samplerate<=15:
                        payload.seek(0)
                        audio,rate=sf.read(payload,dtype='float32')
                        yield qid,audio,rate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/spoken-gsm-original'))
    p.add_argument('--output',type=Path,default=Path('speed-listen5'))
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'audio').mkdir(exist_ok=True)
    rates=[1.]+[i/10 for i in range(5,21) if i!=10]
    cards=[]
    records=[]
    clips=original_clips(args.source)
    try:
        for index,(qid,audio,sr) in enumerate(clips,1):
            if audio.ndim==2:
                audio=audio.mean(axis=1)
            if sr!=16000:
                audio=librosa.resample(audio,orig_sr=sr,target_sr=16000)
            players=[]
            for rate in rates:
                changed=audio if rate==1 else librosa.effects.time_stretch(audio,rate=rate)
                relative=f'audio/{qid}__{rate:g}x.wav'
                sf.write(args.output/relative,changed,16000,subtype='FLOAT')
                records.append(dict(question_id=qid,rate=rate,audio_path=relative,duration_seconds=len(changed)/16000))
                players.append(f'<div><h3>{rate:g}×</h3><audio controls preload="none" src="{relative}"></audio>'
                               f'<p>{len(changed)/16000:.2f} seconds</p></div>')
            cards.append(f'<section><h2>{escape(qid)}</h2>'+''.join(players)+'</section>')
            print(f'[{index}/5] {qid}: 16 speeds written.',flush=True)
            if index==5:
                break
    finally:
        clips.close()
    if len(cards)!=5:
        raise RuntimeError(f'Only {len(cards)} clips <=15 seconds found; need five')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Audio speed listening check</title>'
        '<style>body{font:16px system-ui;max-width:1000px;margin:30px auto;padding:0 20px}'
        'section{border-top:1px solid #ccc;margin-top:30px}audio{width:100%}p{color:#526070}</style>'
        '<h1>Five questions at every speed</h1><p>1× uses unchanged audio after the shared mono/16-kHz preprocessing. '
        'Other versions use librosa time stretching at the same sample rate. Compare voice pitch, words and artifacts. '
        'These are new listening samples, not archived inputs from the earlier accuracy run.</p>'+''.join(cards))
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    archive=args.output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in [args.output/'index.html',args.output/'manifest.jsonl',
                     *(args.output/r['audio_path'] for r in records)]:
            z.write(path,arcname=str(Path(args.output.name)/path.relative_to(args.output)))
    print(f'Finished: 80 clips, no model inference. Download {archive}, extract, and open index.html.',flush=True)


if __name__=='__main__':
    main()
