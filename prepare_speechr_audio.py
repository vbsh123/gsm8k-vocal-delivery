"""Download official SpeechR audio and unpack its nested ZIPs, preserving existing files."""

import argparse
import io
from pathlib import Path
from urllib.request import urlopen
import zipfile


def unpack(archive,output):
    written=0
    for entry in archive.infolist():
        if entry.is_dir() or '__MACOSX' in Path(entry.filename).parts or Path(entry.filename).name.startswith('._'):
            continue
        if entry.filename.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(archive.read(entry))) as nested:
                written+=unpack(nested,output)
        elif Path(entry.filename).suffix.lower() in ('.wav','.flac','.mp3','.ogg'):
            target=(output/entry.filename).resolve()
            if not target.is_relative_to(output.resolve()):
                raise ValueError('Archive path escapes the SpeechR output folder')
            if target.exists():
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix('.tmp')
            temporary.write_bytes(archive.read(entry))
            temporary.replace(target)
            written+=1
    return written


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/speechr'))
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'audio-ready.json').exists():
        print('SpeechR preparation already completed.',flush=True)
        return
    archive=args.output/'speechr.zip'
    if not archive.exists():
        import gdown
        temporary=archive.with_suffix('.download')
        result=gdown.download(id='1BH2r2idILwUHX0NKsXz6GsSXdO0qWly8',output=str(temporary),quiet=False)
        if not result or not zipfile.is_zipfile(temporary):
            raise ValueError('SpeechR download did not produce a valid ZIP')
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as zipped:
        count=unpack(zipped,args.output)
    metadata=args.output/'speechr_acoustic-feature.jsonl'
    if not metadata.exists():
        metadata.write_bytes(urlopen('https://raw.githubusercontent.com/Yanda95/SpeechR/main/speechr_acoustic-feature.jsonl',timeout=60).read())
    audios=[f for f in args.output.rglob('*.wav') if '__MACOSX' not in f.parts]
    if not audios:
        raise ValueError('SpeechR archive contains no audio')
    import json
    (args.output/'audio-ready.json').write_text(json.dumps(dict(wav_files=len(audios),new_files=count))+'\n')
    print(f'SpeechR ready: {len(audios)} WAV files; {count} newly unpacked.',flush=True)


if __name__=='__main__':
    main()
