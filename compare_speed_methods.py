"""Compare tempo algorithms on the same listening recordings; CPU only."""

import argparse
from html import escape
import json
from importlib.metadata import version
from pathlib import Path
import shutil
import subprocess
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('speed-listen5'))
    parser.add_argument('--output',type=Path,default=Path('speed-method-comparison-v2'))
    parser.add_argument('--rates',type=float,nargs='+',default=[.5,.7,.9,1.1,1.5,2.])
    parser.add_argument('--methods',nargs='+',choices=['atempo','rubberband','audiostretchy','librosa'],
                        default=['atempo','rubberband','audiostretchy'])
    args=parser.parse_args()
    if any(not .5<=rate<=2 for rate in args.rates):
        parser.error('Choose rates between 0.5 and 2.')
    import imageio_ffmpeg
    import soundfile as sf
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    if 'rubberband' in args.methods:
        filters=subprocess.check_output([ffmpeg,'-hide_banner','-filters'],text=True,stderr=subprocess.DEVNULL)
        if 'rubberband' not in filters:
            parser.error('FFmpeg lacks Rubber Band. Install imageio-ffmpeg==0.8.1 and unset IMAGEIO_FFMPEG_EXE if overridden.')
    if 'audiostretchy' in args.methods:
        from audiostretchy.core import stretch_audio
    records={}
    for line in (args.source/'manifest.jsonl').read_text().splitlines():
        row=json.loads(line)
        records.setdefault(row['question_id'],{})[row['rate']]=row
    audio_dir=args.output/'audio'
    audio_dir.mkdir(parents=True,exist_ok=True)
    sections=[]
    manifest=[]
    for qid,versions in records.items():
        original=args.source/versions[1.]['audio_path']
        original_name=f'audio/{qid}__original.wav'
        shutil.copyfile(original,args.output/original_name)
        manifest.append(dict(question_id=qid,method='original',rate=1.,audio_path=original_name,
                             duration_seconds=sf.info(original).duration))
        players=[f'<h3>Unchanged original: 1×</h3><audio controls src="{escape(original_name,quote=True)}"></audio>']
        for rate in args.rates:
            for method in args.methods:
                name=f'audio/{qid}__{rate:g}x__{method}.wav'
                destination=args.output/name
                if method=='librosa':
                    shutil.copyfile(args.source/versions[rate]['audio_path'],destination)
                elif method=='audiostretchy':
                    stretch_audio(original,destination,ratio=1/rate,gap_ratio=1/rate)
                else:
                    audio_filter=f'atempo={rate:g}' if method=='atempo' else f'rubberband=tempo={rate:g}:pitch=1'
                    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y',
                                    '-i',str(original),'-af',audio_filter,
                                    '-c:a','pcm_f32le',str(destination)],check=True)
                seconds=sf.info(destination).duration
                manifest.append(dict(question_id=qid,method=method,rate=rate,audio_path=name,duration_seconds=seconds))
                players.append(f'<h3>{rate:g}× — {method} ({seconds:.2f}s)</h3><audio controls preload="none" src="{escape(name,quote=True)}"></audio>')
        sections.append(f'<section><h2>{escape(qid)}</h2>'+''.join(players)+'</section>')
        print(f'{qid}: original and {len(args.methods)} methods ready.',flush=True)
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Compare speed methods</title>'
        '<style>body{font:16px system-ui;max-width:900px;margin:30px auto;padding:20px}'
        'audio{width:100%}section{border-top:1px solid #ccc;margin-top:30px}</style>'
        '<h1>Same recording, different speed-changing methods</h1>'
        '<p>Compare each method with the unchanged 1× recording. Rates below 1 slow speech; rates above 1 speed it up. '
        'Rubber Band uses FFmpeg with pitch=1. AudioStretchy uses duration ratio=1/rate for speech and gaps. '
        'No method guarantees artifact-free speech. No accuracy evaluation was run.</p>'+''.join(sections))
    (args.output/'recipe.json').write_text(json.dumps(dict(
        source=str(args.source.resolve()),rates=args.rates,
        ffmpeg_version=subprocess.check_output([ffmpeg,'-version'],text=True).splitlines()[0],
        methods=args.methods,rubberband_filter='rubberband=tempo=RATE:pitch=1',
        audiostretchy_version=version('audiostretchy') if 'audiostretchy' in args.methods else None,
        audiostretchy_ratio='1/rate; gap_ratio=1/rate',codec='method defaults; FFmpeg outputs pcm_f32le'),indent=2)+'\n')
    archive=args.output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in args.output.rglob('*'):
            if path.is_file():
                zipped.write(path,Path(args.output.name)/path.relative_to(args.output))
    print(f'Download {archive}, extract, and open index.html.',flush=True)


if __name__=='__main__':
    main()
