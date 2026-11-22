"""Size all saved experiment audio before creating an archive. No audio is changed."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil

AUDIO = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.opus'}


def inventory(root, hash_contents=False):
    groups = defaultdict(lambda: dict(files=0, bytes=0))
    targets = {}
    files = []
    errors = []
    for directory, names, filenames in os.walk(root):
        names[:] = sorted(n for n in names if not n.startswith(('.venv', 'models--', 'datasets--', '.git')))
        for name in sorted(filenames):
            path = Path(directory) / name
            if path.suffix.lower() not in AUDIO:
                continue
            try:
                real = path.resolve(strict=True)
                size = real.stat().st_size
                if not real.is_file():
                    continue
                relative = path.relative_to(root)
                group = relative.parts[0] if len(relative.parts)>1 else '(root)'
                groups[group]['files'] += 1
                groups[group]['bytes'] += size
                files.append(dict(path=str(path), resolved_path=str(real), bytes=size))
                targets[str(real)] = size
            except OSError as exc:
                errors.append(dict(path=str(path), error=str(exc)))
    result = dict(root=str(root.resolve()), audio_files=len(files),
                  logical_bytes=sum(r['bytes'] for r in files), unique_target_files=len(targets),
                  unique_target_bytes=sum(targets.values()), folders=dict(groups), errors=errors,
                  free_bytes=shutil.disk_usage(root).free, files=files)
    if hash_contents:
        hashes = {}
        for index, (path, size) in enumerate(sorted(targets.items()), 1):
            digest = hashlib.sha256()
            with Path(path).open('rb') as audio:
                for block in iter(lambda: audio.read(1024*1024), b''):
                    digest.update(block)
            hashes.setdefault(digest.hexdigest(), size)
            if index % 1000 == 0:
                print(f'Hashing audio: {index}/{len(targets)} files.', flush=True)
        result.update(unique_content_files=len(hashes), unique_content_bytes=sum(hashes.values()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs'))
    parser.add_argument('--hash', action='store_true', help='Also read all audio to measure byte-identical duplication.')
    parser.add_argument('--output', type=Path, default=Path('outputs/all-audio-size.json'))
    args = parser.parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    result = inventory(args.root, args.hash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'{"Folder":58} {"Files":>8} {"GiB":>10}')
    for name, values in sorted(result['folders'].items(), key=lambda item: -item[1]['bytes']):
        print(f'{name:58} {values["files"]:8} {values["bytes"]/2**30:10.3f}')
    print(f'\nAll audio paths: {result["audio_files"]:,}; uncompressed total: {result["logical_bytes"]/2**30:.3f} GiB')
    print(f'Unique resolved files: {result["unique_target_files"]:,}; size: {result["unique_target_bytes"]/2**30:.3f} GiB')
    if args.hash:
        print(f'Unique audio contents: {result["unique_content_files"]:,}; size: {result["unique_content_bytes"]/2**30:.3f} GiB')
    print(f'Free space on source filesystem: {result["free_bytes"]/2**30:.3f} GiB')
    print(f'Unreadable/broken paths: {len(result["errors"])}; full inventory: {args.output}')
    print('All audio types/styles/voices under the scan root are included, regardless of emotion classification.')
    print('ZIP size is not measured yet; compressed size requires creating an archive or sampling compression.')


if __name__ == '__main__':
    main()
