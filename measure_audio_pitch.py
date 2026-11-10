"""Measure voiced pitch on Azure recordings and compare matched deliveries; CPU only."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
import time

from scoring import score_response
from steer import read_rows

PITCH_SETTINGS=dict(sample_rate=16000,method='librosa.pyin',fmin=50,fmax=800,
                    frame_length=2048,hop_length=320,center=True)


def measure_pitch(audio, librosa, np):
    f0,voiced,_=librosa.pyin(audio,sr=16000,fmin=50,fmax=800,
                                     frame_length=2048,hop_length=320,center=True)
    valid=voiced & np.isfinite(f0) & (f0>0)
    trace=np.where(valid,f0,np.nan)
    values=trace[valid]
    stats=dict(duration_seconds=len(audio)/16000,frames=len(f0),voiced_frames=int(valid.sum()),
               voiced_fraction=float(valid.mean()),mean_f0_hz=None,median_f0_hz=None,
               p10_f0_hz=None,p90_f0_hz=None,pitch_span_semitones=None)
    if len(values):
        low,high=np.percentile(values,[10,90])
        stats.update(mean_f0_hz=float(values.mean()),median_f0_hz=float(np.median(values)),
                     p10_f0_hz=float(low),p90_f0_hz=float(high),
                     pitch_span_semitones=float(12*np.log2(high/low)))
    return stats,trace


def load_audio(path, sf, librosa):
    audio,rate=sf.read(path,dtype='float32')
    if audio.ndim==2:
        audio=audio.mean(axis=1)
    if rate!=16000:
        audio=librosa.resample(audio,orig_sr=rate,target_sr=16000)
    if not len(audio):
        raise ValueError(f'Empty audio: {path}')
    return audio


def freeze_settings(output, settings):
    output.mkdir(parents=True,exist_ok=True)
    path=output/'pitch-settings.json'
    if path.exists() and json.loads(path.read_text())!=settings:
        raise ValueError('Pitch settings changed; use a separate output folder')
    if not path.exists():
        path.write_text(json.dumps(settings,indent=2)+'\n')


def summarize(rows, output):
    rows=list({r['audio_file']:r for r in rows}.values())
    groups=defaultdict(list)
    for r in rows:
        groups[r['style']].append(r)
    print('\nVoiced pitch by requested delivery (unpaired; each recording gets equal weight)')
    print(f'{"Delivery":12} {"Clips":>6} {"With F0":>8} {"Mean Hz":>9} {"Median Hz":>10} {"Span st":>9} {"Voiced":>8}')
    for style,items in sorted(groups.items()):
        valid=[r for r in items if r['median_f0_hz'] is not None]
        if valid:
            print(f'{style:12} {len(items):6} {len(valid):8} {mean(r["mean_f0_hz"] for r in valid):9.1f} '
                  f'{median(r["median_f0_hz"] for r in valid):10.1f} '
                  f'{mean(r["pitch_span_semitones"] for r in valid):9.2f} {mean(r["voiced_fraction"] for r in items):8.1%}')
        else:
            print(f'{style:12} {len(items):6} {0:8}    no voiced pitch detected')
    neutral={(r['question_id'],r['voice']):r for r in rows if r['style']=='neutral' and r['median_f0_hz'] is not None}
    comparisons=[]
    print('\nMatched question + voice versus neutral')
    print(f'{"Delivery":12} {"Pairs":>6} {"Mean Δ Hz":>10} {"Mean Δ st":>10} {"Higher":>8}')
    for style in sorted(set(groups)-{'neutral'}):
        pairs=[(r,neutral[r['question_id'],r['voice']]) for r in groups[style]
               if r['median_f0_hz'] is not None and (r['question_id'],r['voice']) in neutral]
        if not pairs:
            continue
        shifts=[12*math.log2(r['median_f0_hz']/b['median_f0_hz']) for r,b in pairs]
        row=dict(style=style,pairs=len(pairs),mean_f0_change_hz=mean(r['mean_f0_hz']-b['mean_f0_hz'] for r,b in pairs),
                 mean_median_pitch_change_semitones=mean(shifts),fraction_higher=mean(v>0 for v in shifts))
        comparisons.append(row)
        print(f'{style:12} {len(pairs):6} {row["mean_f0_change_hz"]:+10.1f} {mean(shifts):+10.2f} {row["fraction_higher"]:8.1%}')
    print('Mean Hz averages per-clip voiced means; Median Hz is the median of per-clip medians.')
    print('Span is voiced P90/P10 in semitones. Unvoiced frames are excluded, not counted as 0 Hz.')
    print('Pitch estimates can have octave/voicing errors. These associations do not isolate a causal pitch effect.')
    (output/'summary.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    with (output/'pitch.csv').open('w',newline='') as stream:
        if rows:
            fields=list(dict.fromkeys(k for r in rows for k in r))
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',type=Path,nargs='+',default=[Path('outputs/azure-jenny500-extra'),
        Path('outputs/azure-jenny500-neutral'),Path('outputs/azure-jenny-next500')])
    p.add_argument('--output',type=Path,default=Path('outputs/jenny-pitch'))
    p.add_argument('--limit-questions',type=int,help='First N question/voice pairs; omit to measure all.')
    p.add_argument('--evaluated-only',action='store_true',help='Use only clips with completed DIFFA answers.')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    path=args.output/'results-pitch.jsonl'
    if args.summary_only:
        summarize(read_rows(path),args.output)
        return
    import librosa
    import numpy as np
    import soundfile as sf
    tasks={}
    for folder in args.paths:
        answers={r['audio_path']:r for r in read_rows(folder/'results-matched-accuracy.jsonl')}
        for task in read_rows(folder/'plan.jsonl'):
            if task['provider']!='azure' or (args.evaluated_only and task['audio_path'] not in answers):
                continue
            key=task['question_id'],task['voice'],task['style']
            record=dict(question_id=task['question_id'],voice=task['voice'],style=task['style'],
                        audio_file=str((folder/task['audio_path']).resolve()),correct=None)
            if key in tasks and tasks[key]['audio_file']!=record['audio_file']:
                raise ValueError(f'Duplicate question/voice/style in input folders: {key}')
            if task['audio_path'] in answers:
                result=answers[task['audio_path']]
                record['correct']=score_response(result['response'],result['reference_answer'])['correct']
            tasks[key]=record
    keys=sorted({key[:2] for key in tasks})
    if args.limit_questions is not None:
        if args.limit_questions<1:
            p.error('--limit-questions must be positive')
        keys=keys[:args.limit_questions]
    keys=set(keys)
    selected=[r for key,r in sorted(tasks.items()) if key[:2] in keys]
    freeze_settings(args.output,dict(**PITCH_SETTINGS,librosa=librosa.__version__,
                    paths=[str(f.resolve()) for f in args.paths],evaluated_only=args.evaluated_only))
    rows=read_rows(path,repair=True)
    saved={r['audio_file']:r for r in rows}
    for index,record in enumerate(selected,1):
        audio_path=Path(record['audio_file'])
        if not audio_path.exists():
            print('Missing audio:',audio_path,flush=True)
            continue
        stat=audio_path.stat()
        if record['audio_file'] in saved:
            old=saved[record['audio_file']]
            if (old['audio_bytes'],old['audio_mtime_ns'])!=(stat.st_size,stat.st_mtime_ns):
                raise ValueError(f'Measured audio changed: {audio_path}')
            continue
        start=time.monotonic()
        print(f'[{index}/{len(selected)}] {record["question_id"]} {record["style"]}: estimating voiced pitch on CPU',flush=True)
        stats,_=measure_pitch(load_audio(audio_path,sf,librosa),librosa,np)
        row=dict(**record,**stats,audio_bytes=stat.st_size,audio_mtime_ns=stat.st_mtime_ns)
        with path.open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        rows.append(row)
        print(f'[{index}/{len(selected)}] {record["question_id"]} {record["style"]}: '
              f'median F0={row["median_f0_hz"]}; {time.monotonic()-start:.1f}s CPU',flush=True)
    summarize(rows,args.output)


if __name__=='__main__':
    main()
