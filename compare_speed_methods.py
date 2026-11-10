"""Compare FFmpeg atempo and existing librosa listening samples; CPU only."""

import argparse
from html import escape
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('speed-listen5'))
    parser.add_argument('--output',type=Path,default=Path('speed-method-comparison'))
    parser.add_argument('--rates',type=float,nargs='+',default=[.9,1.1,1.3])
    args=parser.parse_args()
    if any(not .5<=rate<=2 for rate in args.rates):
        parser.error('Choose rates between 0.5 and 2.')
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
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
        players=[f'<h3>Unchanged original: 1×</h3><audio controls src="{escape(original_name,quote=True)}"></audio>']
        for rate in args.rates:
            for method in ['librosa','atempo']:
                name=f'audio/{qid}__{rate:g}x__{method}.wav'
                destination=args.output/name
                if method=='librosa':
                    shutil.copyfile(args.source/versions[rate]['audio_path'],destination)
                else:
                    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y',
                                    '-i',str(original),'-af',f'atempo={rate:g}',
                                    '-c:a','pcm_f32le',str(destination)],check=True)
                manifest.append(dict(question_id=qid,method=method,rate=rate,audio_path=name))
                players.append(f'<h3>{rate:g}× — {method}</h3><audio controls preload="none" src="{escape(name,quote=True)}"></audio>')
        sections.append(f'<section><h2>{escape(qid)}</h2>'+''.join(players)+'</section>')
        print(f'{qid}: original and both methods ready.',flush=True)
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Compare speed methods</title>'
        '<style>body{font:16px system-ui;max-width:900px;margin:30px auto;padding:20px}'
        'audio{width:100%}section{border-top:1px solid #ccc;margin-top:30px}</style>'
        '<h1>Same recording, two speed-changing methods</h1>'
        '<p>Compare the unchanged 1× recording, existing librosa samples, and FFmpeg atempo. '
        'Neither method guarantees artifact-free speech. No accuracy evaluation was run.</p>'+''.join(sections))
    (args.output/'recipe.json').write_text(json.dumps(dict(
        source=str(args.source.resolve()),rates=args.rates,
        ffmpeg_version=subprocess.check_output([ffmpeg,'-version'],text=True).splitlines()[0],
        filter='atempo',codec='pcm_f32le'),indent=2)+'\n')
    archive=args.output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in args.output.rglob('*'):
            if path.is_file():
                zipped.write(path,Path(args.output.name)/path.relative_to(args.output))
    print(f'Download {archive}, extract, and open index.html.',flush=True)


if __name__=='__main__':
    main()
