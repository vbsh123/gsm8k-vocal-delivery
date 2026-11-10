"""Reconstruct a saved speed sweep for listening and pitch checks; no model or GPU."""

import argparse
from html import escape
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

from measure_audio_pitch import PITCH_SETTINGS, freeze_settings, load_audio, measure_pitch
from steer import read_rows


def contour_difference(reference, changed, rate, np):
    indices=np.rint(np.arange(len(changed))*rate).astype(int)
    inside=indices<len(reference)
    values=changed[inside]
    base=reference[indices[inside]]
    valid=np.isfinite(values)&np.isfinite(base)&(values>0)&(base>0)
    shift=12*np.log2(values[valid]/base[valid])
    return dict(aligned_voiced_frames=len(shift),
                median_absolute_contour_shift_semitones=float(np.median(np.abs(shift))) if len(shift) else None)


def report(rows, output):
    latest={(r['clip_id'],r['rate']):r for r in rows}
    print('\nSpeed pitch audit: matched against the same unchanged recording')
    print(f'{"Rate":>6} {"Measured":>9} {"With F0":>8} {"Mean Δ Hz":>10} {"Mean Δ st":>10} {"Contour |Δ| st":>15}')
    summary=[]
    for rate in sorted({r['rate'] for r in latest.values()}):
        measured=[r for r in latest.values() if r['rate']==rate]
        pairs=[(r,latest[r['clip_id'],1.]) for r in measured if (r['clip_id'],1.) in latest
               and r['median_f0_hz'] is not None and latest[r['clip_id'],1.]['median_f0_hz'] is not None]
        if not pairs:
            print(f'{rate:5g}x {len(measured):9} {0:8}    no usable paired F0')
            continue
        shifts=[12*math.log2(r['median_f0_hz']/b['median_f0_hz']) for r,b in pairs]
        contour=[r['median_absolute_contour_shift_semitones'] for r,b in pairs
                 if r['median_absolute_contour_shift_semitones'] is not None]
        row=dict(rate=rate,measured=len(measured),paired=len(pairs),
                 mean_f0_change_hz=mean(r['mean_f0_hz']-b['mean_f0_hz'] for r,b in pairs),
                 mean_median_pitch_change_semitones=mean(shifts),
                 median_absolute_contour_shift_semitones=median(contour) if contour else None)
        summary.append(row)
        error=f'{median(contour):.2f}' if contour else 'n/a'
        print(f'{rate:5g}x {len(measured):9} {len(pairs):8} {row["mean_f0_change_hz"]:+10.2f} {mean(shifts):+10.3f} {error:>15}')
    print('Δ st compares per-recording median F0. Contour error approximately aligns time using the known rate.')
    print('Small global shifts do not exclude local artifacts. F0 tracking can itself make octave/voicing errors.')
    print('Saved inference waveforms are used where available; otherwise the recorded transform is reconstructed.')
    print('Reconstruction checks the recorded pipeline, not an archived waveform that was never saved. Listen too.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


def listening_page(records, rates, rows, output, count):
    latest={(r['clip_id'],r['rate']):r for r in rows}
    cards=[]
    for record in records[:count]:
        players=[]
        for rate in [1.]+[r for r in rates if r!=1]:
            row=latest.get((record['clip_id'],rate))
            name=f'audio/{record["clip_id"]}__{rate:g}x.wav'
            if not row or not (output/name).exists():
                continue
            f0=f'{row["median_f0_hz"]:.1f} Hz' if row['median_f0_hz'] is not None else 'no voiced F0'
            players.append(f'<div><h3>{rate:g}×</h3><audio controls preload="none" src="{escape(name,quote=True)}"></audio>'
                           f'<p>{row["duration_seconds"]:.2f}s; median F0 {f0}; {escape(row["waveform_source"])}</p></div>')
        cards.append(f'<section><h2>{escape(record["question_id"])}</h2>'+''.join(players)+'</section>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Speed and pitch audit</title>'
        '<style>body{font:16px system-ui;max-width:1050px;margin:30px auto;padding:0 20px}section{border-top:1px solid #ccc;margin-top:30px}'
        'audio{width:100%}p{color:#526070}</style><h1>Speed and pitch listening check</h1>'
        '<p>Compare the same question at 1× and altered speeds. Listen for voice pitch, words, and metallic or smeared artifacts. '
        'Players show saved inputs when available; other clips are reconstructions using the recorded library version. '
        'F0 estimates are supporting measurements, not proof of perceptual equivalence.</p>'+''.join(cards))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--speed-run',type=Path,default=Path('outputs/spoken-gsm-speed100'))
    p.add_argument('--output',type=Path,default=Path('outputs/spoken-gsm-speed100-pitch-audit'))
    p.add_argument('--num-examples',type=int,default=3,help='Audit first N saved recordings; increase to 100 to check the whole old sweep.')
    p.add_argument('--listen-examples',type=int,default=3,help='Save listening waveforms for the first N audited recordings.')
    p.add_argument('--summary-only',action='store_true')
    args=p.parse_args()
    results=args.output/'results-pitch.jsonl'
    if args.summary_only:
        report(read_rows(results),args.output)
        return
    import librosa
    import numpy as np
    import soundfile as sf
    config_path=args.speed_run/'run_config.json'
    config=json.loads(config_path.read_text())
    if config['settings']['librosa']!=librosa.__version__:
        p.error(f'Use librosa {config["settings"]["librosa"]} to reconstruct the recorded transform exactly')
    if config['settings']['transform']!='librosa.effects.time_stretch; rate1 bypassed':
        p.error('Unsupported saved audio transform')
    if not 1<=args.num_examples<=len(config['recordings']) or args.listen_examples<0:
        p.error('Choose a positive available question count and nonnegative listening count')
    freeze_settings(args.output,dict(**PITCH_SETTINGS,librosa=librosa.__version__,
                    speed_run=str(args.speed_run.resolve()),source_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest()))
    (args.output/'audio').mkdir(exist_ok=True)
    rows=read_rows(results,repair=True)
    latest={(r['clip_id'],r['rate']):r for r in rows}
    records=config['recordings'][:args.num_examples]
    rates=[1.]+[r for r in config['rates'] if r!=1]
    for index,record in enumerate(records):
        listen=index<args.listen_examples
        cid=record['clip_id']
        if all((cid,r) in latest and (not listen or (args.output/'audio'/f'{cid}__{r:g}x.wav').exists()) for r in rates):
            continue
        path=Path(record['audio_path'])
        stat=path.stat()
        if (record['audio_bytes'],record['audio_mtime_ns'])!=(stat.st_size,stat.st_mtime_ns):
            raise ValueError(f'Original waveform changed since the speed experiment: {path}')
        original=load_audio(path,sf,librosa)
        trace_path=args.output/f'baseline-pitch-{index:04d}.npy'
        base_trace=np.load(trace_path) if trace_path.exists() else None
        for rate in rates:
            target=args.output/'audio'/f'{cid}__{rate:g}x.wav'
            key=(cid,rate)
            if key in latest and (not listen or target.exists()) and (rate!=1 or base_trace is not None):
                continue
            saved=args.speed_run/'audio'/target.name
            if saved.exists():
                audio=load_audio(saved,sf,librosa)
                source='saved inference input'
            else:
                audio=original if rate==1 else librosa.effects.time_stretch(original,rate=rate)
                source='reconstructed from recorded recipe'
            if key not in latest or (rate==1 and base_trace is None):
                print(f'[{index+1}/{len(records)}] {record["question_id"]} {rate:g}x: estimating voiced pitch on CPU',flush=True)
                stats,trace=measure_pitch(audio,librosa,np)
                if rate==1:
                    base_trace=trace
                    temporary=trace_path.with_suffix('.tmp')
                    with temporary.open('wb') as stream:
                        np.save(stream,trace)
                    temporary.replace(trace_path)
                row=dict(clip_id=cid,question_id=record['question_id'],rate=rate,waveform_source=source,
                         **stats,**contour_difference(base_trace,trace,rate,np))
                with results.open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                rows.append(row)
                latest[key]=row
            if listen and not target.exists():
                sf.write(target,audio,16000,subtype='FLOAT')
    report(rows,args.output)
    listening_page(records,config['rates'],rows,args.output,args.listen_examples)
    print(f'Download the whole folder and open index.html: {args.output}',flush=True)


if __name__=='__main__':
    main()
