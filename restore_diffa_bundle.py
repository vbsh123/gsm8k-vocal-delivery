"""Restore a Jenny audio or DIFFA migration ZIP into the cloned repo without overwriting different files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def restore(path, root):
    root = root.resolve()
    checksum = path.with_suffix('.zip.sha256')
    if checksum.is_file():
        expected = checksum.read_text().split()[0]
        if digest(path) != expected:
            raise ValueError(f'Archive checksum mismatch: {path}')
        print(f'Archive checksum verified: {path.name}', flush=True)
    else:
        print(f'No sidecar checksum for {path.name}; ZIP CRCs will still be checked while extracting.', flush=True)
    restored = reused = 0
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if 'DIFFA_MIGRATION_MANIFEST.json' in names:
            manifest = json.loads(archive.read('DIFFA_MIGRATION_MANIFEST.json'))
            if manifest['repo'] != str(root):
                raise ValueError(f'Frozen paths require clone at {manifest["repo"]}; current root is {root}.')
        elif 'JENNY_AUDIO_MANIFEST.json' not in names:
            raise ValueError('Not a recognized Jenny audio or DIFFA migration bundle.')
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if (member.is_absolute() or '..' in member.parts
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError(f'Unsafe archive member: {info.filename}')
            destination = root.joinpath(*member.parts)
            if not destination.resolve().is_relative_to(root):
                raise ValueError(f'Archive member escapes destination: {info.filename}')
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name+'.restore-tmp')
            result = hashlib.sha256()
            try:
                with archive.open(info) as source, temporary.open('xb') as target:
                    for block in iter(lambda: source.read(4*1024**2), b''):
                        target.write(block)
                        result.update(block)
                if destination.exists():
                    if digest(destination) != result.hexdigest():
                        raise ValueError(f'Existing file differs; left unchanged: {destination}')
                    reused += 1
                else:
                    temporary.rename(destination)
                    restored += 1
            finally:
                temporary.unlink(missing_ok=True)
            if (restored+reused) % 1000 == 0:
                print(f'Extracted/verified {restored+reused:,}/{len(names):,} files.', flush=True)
    print(f'Restored {restored:,}; already identical {reused:,}: {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for path in args.archives:
        restore(path.resolve(), root)


if __name__ == '__main__':
    main()
