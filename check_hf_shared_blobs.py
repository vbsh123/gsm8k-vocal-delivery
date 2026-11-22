"""Read-only reference audit of nonstandard shared hub/blobs storage. Never deletes files."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path


def audit(hub):
    hub = hub.resolve()
    shared = hub/'blobs'
    files = {}
    inodes = defaultdict(set)
    for directory, _, names in os.walk(shared):
        for name in names:
            path = Path(directory)/name
            if path.is_symlink() or not path.is_file():
                continue
            stat = path.stat()
            absolute = str(path.resolve())
            files[absolute] = dict(path=absolute, bytes=stat.st_size, hardlink_count=stat.st_nlink, owners=[])
            inodes[stat.st_dev, stat.st_ino].add(absolute)
    owners = defaultdict(set)
    for directory, names, filenames in os.walk(hub, followlinks=False):
        names[:] = [n for n in names if (Path(directory)/n).resolve() != shared]
        for name in [*names, *filenames]:
            path = Path(directory)/name
            try:
                owner = path.relative_to(hub).parts[0]
                if path.is_symlink():
                    target = path.resolve()
                    if str(target) in files:
                        owners[str(target)].add(owner)
                    elif target.is_dir() and target.is_relative_to(shared):
                        prefix = str(target)+'/'
                        for key in files:
                            if key.startswith(prefix):
                                owners[key].add(owner)
                elif path.is_file():
                    stat = path.stat()
                    for key in inodes.get((stat.st_dev, stat.st_ino), ()):
                        owners[key].add(owner)
            except (OSError, RuntimeError):
                continue
    groups = defaultdict(lambda: dict(files=0, bytes=0))
    for key, info in files.items():
        info['owners'] = sorted(owners[key])
        group = ', '.join(info['owners']) if info['owners'] else 'NO REFERENCE FOUND IN HUB (inspect before deleting)'
        groups[group]['files'] += 1
        groups[group]['bytes'] += info['bytes']
    return dict(hub=str(hub), files=list(files.values()), groups=dict(groups),
                note='Only symlink/hardlink references inside this hub are inspected. No-reference status is not proof that external configs or running jobs do not use a file.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hub', type=Path, default=Path('/workspace/.hf_home/hub'))
    parser.add_argument('--output', type=Path, default=Path('outputs/shared-hf-blobs-check.json'))
    args = parser.parse_args()
    result = audit(args.hub)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Shared hub/blobs reference audit; nothing deleted.')
    for name, values in sorted(result['groups'].items(), key=lambda item: -item[1]['bytes']):
        print(f'{values["bytes"]/2**30:8.2f} GiB {values["files"]:7} files  {name}')
    print('\nLargest 10 shared files:')
    for info in sorted(result['files'], key=lambda row: -row['bytes'])[:10]:
        print(f'{info["bytes"]/2**30:.2f} GiB links={info["hardlink_count"]} {info["path"]} owners={info["owners"]}')
    print(f'Full details: {args.output}')


if __name__ == '__main__':
    main()
