"""Verify the old GSM WAV batches against the local six-dataset Jenny backup."""
import argparse
import hashlib
import stat
from pathlib import Path

batches = [
    ('azure-jenny-next500', 550, 1049, 'angry happy neutral sad', 'bb58862b80719b3ddb3e4ea83906f0ce21db282c53914bc832fb754e87b97659'),
    ('azure-jenny500-extra', 50, 549, 'angry happy sad', 'ab082bf42c49f4f3aa170f4ab12e13b12955486eaaf03e8d4db7467105de7a7b'),
    ('azure-jenny-remaining-test', 1050, 1318, 'angry happy neutral sad', '646eb2eeb80c3c89c0e2152d7640497ebf2f7a7fe97e4c4a601b945ad1498dc9'),
]

def signature(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise OSError(f'Expected a regular WAV at {path}')
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delete', action='store_true', help='remove WAVs from verified batches')
    args = parser.parse_args()
    root = Path('outputs')
    if root.is_symlink() or not root.is_dir():
        parser.error('Run from the experiment repo containing outputs/.')
    for directory, start, end, styles, expected in batches:
        folder = root / directory
        if folder.is_symlink():
            print(f'Skipping symlink {folder}')
            continue
        combined = hashlib.sha256()
        checked = {}
        try:
            for i in range(start, end+1):
                for style in styles.split():
                    p = folder / f'q{i:05d}-azure-{style}.wav'
                    checked[p] = signature(p)
                    h = hashlib.sha256()
                    with p.open('rb') as stream:
                        for chunk in iter(lambda: stream.read(1024*1024), b''):
                            h.update(chunk)
                    combined.update(f'{h.hexdigest()}  {p.name}\n'.encode())
            if combined.hexdigest() != expected or any(signature(p) != old for p, old in checked.items()):
                print(f'Skipping {folder}: files changed or batch differs from backup')
                continue
        except OSError as ex:
            print(f'Skipping {folder}: {ex}')
            continue
        if args.delete:
            for p, old in checked.items():
                if signature(p) != old:
                    raise SystemExit(f'Stopped: {p} changed since verification')
                p.unlink()
        size = sum(old[2] for old in checked.values()) / 2**30
        print(f'{"Removed" if args.delete else "Verified"} {len(checked)} WAVs from {folder} ({size:.2f} GiB)')
if __name__ == '__main__':
    main()
