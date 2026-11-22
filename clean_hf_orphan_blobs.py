"""Preview/remove large shared HF cache blobs with no remaining references. No experiment files touched."""
import argparse
import os
from pathlib import Path

from check_hf_shared_blobs import audit


def external_references(workspace, hub, candidates):
    blocked = {}
    for directory, names, filenames in os.walk(workspace, followlinks=False):
        names[:] = [n for n in names if (Path(directory)/n).resolve() != hub]
        for name in [*names, *filenames]:
            path = Path(directory)/name
            if not path.is_symlink():
                continue
            try:
                target = str(path.resolve())
                for key in candidates:
                    if key == target or key.startswith(target+'/'):
                        blocked[key] = f'external symlink: {path}'
            except (OSError, RuntimeError):
                continue
    # Look for open or memory-mapped weights in processes visible inside this container.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            maps = (process/'maps').read_text(errors='replace')
            for key in candidates:
                if key in maps:
                    blocked[key] = f'memory-mapped by PID {process.name}'
        except (OSError, PermissionError):
            pass
        try:
            for fd in (process/'fd').iterdir():
                try:
                    key = os.readlink(fd)
                    if key in candidates:
                        blocked[key] = f'open by PID {process.name}'
                except OSError:
                    continue
        except (OSError, PermissionError):
            pass
    return blocked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hub', type=Path, default=Path('/workspace/.hf_home/hub'))
    parser.add_argument('--workspace', type=Path, default=Path('/workspace'))
    parser.add_argument('--delete', action='store_true', help='Delete the listed unreferenced cache blobs after fresh checks.')
    args = parser.parse_args()
    hub = args.hub.resolve()
    result = audit(hub)
    candidates = {r['path']: r for r in result['files']
                  if not r['owners'] and r['hardlink_count'] == 1 and r['bytes'] >= 1024**3}
    blocked = external_references(args.workspace.resolve(), hub, candidates)
    selected = [r for key, r in candidates.items() if key not in blocked]
    print('Only shared-cache files >=1 GiB, no remaining hub owner, one hardlink, and no visible open/external symlink reference.')
    print('This is a cache cleanup, not a proof of absence from every historical config. An unexpectedly needed model file can be downloaded again.')
    for r in sorted(selected, key=lambda r: -r['bytes']):
        print(f'{r["bytes"]/2**30:6.2f} GiB {r["path"]}')
    for key, reason in blocked.items():
        print(f'KEEP: {key}: {reason}')
    print(f'Total candidate space: {sum(r["bytes"] for r in selected)/2**30:.2f} GiB', flush=True)
    if not args.delete:
        print('Preview only. Add --delete to remove these cache files.')
        return
    freed = 0
    for r in selected:
        path = Path(r['path'])
        if path.is_symlink() or not path.resolve().is_relative_to(hub/'blobs'):
            raise ValueError(f'Path changed since scan: {path}')
        current = path.stat()
        if current.st_nlink != 1 or current.st_size != r['bytes']:
            raise ValueError(f'File changed since scan; stopped: {path}')
        path.unlink()
        freed += r['bytes']
    print(f'Removed {freed/2**30:.2f} GiB of unreferenced cache blobs. Model-cache directories and experiment outputs were not removed.')


if __name__ == '__main__':
    main()
