"""Stop only the old sequential Phi diagnostic; retain its resumable evidence."""
import os
from pathlib import Path
import signal


def command(pid):
    try: return [os.fsdecode(a) for a in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if a]
    except OSError: return []


def has_script(args, name): return any(Path(a).name == name for a in args)


def stop():
    wrappers=[];workers=[];waiters=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        args=command(path.name)
        if has_script(args,'run_phi4_all_layer_diagnostic.sh'): wrappers.append(int(path.name))
        elif has_script(args,'phi4_all_layer_diagnostic.py') and '--worker-id' not in args:
            mode=args[args.index('--mode')+1] if '--mode' in args else 'collect'
            if mode in ('prepare','select'):
                raise RuntimeError('Sequential diagnostic is writing config/selection; wait for that short CPU stage before switching.')
            if mode in ('collect','steer'): workers.append(int(path.name))
    for pid in wrappers:
        try: children=[int(p) for p in (Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split()]
        except OSError: continue
        for child in children:
            args=command(child)
            if args and Path(args[0]).name=='flock': waiters.append(child)
            elif args and child not in workers:
                raise RuntimeError('Old wrapper has an unexpected active child; no processes stopped.')
    targets=list(dict.fromkeys(wrappers+workers+waiters))
    for pid in targets:
        try: os.kill(pid,signal.SIGTERM)
        except ProcessLookupError: pass
    print('Stopped old sequential diagnostic PIDs:',targets)
    print('Completed caches/results retained. Phi L17 -2, Flamingo and split workers untouched.')


if __name__=='__main__':stop()
