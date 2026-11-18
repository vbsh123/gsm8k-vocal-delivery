"""Release only a split launcher blocked in flock; never stop running workers/Phi."""
import os
from pathlib import Path
import signal


def command(pid):
    try: return [os.fsdecode(a) for a in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if a]
    except OSError: return []


def stop():
    waiting=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        args=command(path.name)
        if not any(Path(a).name=='run_phi4_all_layers_split.sh' for a in args): continue
        pid=int(path.name)
        try: children=[int(p) for p in (path/'task'/path.name/'children').read_text().split()]
        except OSError: continue
        commands=[command(child) for child in children]
        if not children or any(not a or Path(a[0]).name!='flock' for a in commands):
            raise RuntimeError(f'Split launcher{pid} is not just waiting in flock; workers/preparation may have started. No processes stopped.')
        waiting.append((pid,children))
    for pid,children in waiting:
        for target in [pid]+children:
            try: os.kill(target,signal.SIGTERM)
            except ProcessLookupError: pass
        print(f'Stopped waiting split launcher{pid}; Phi -2 and all model workers untouched.')
    if not waiting: print('No waiting split launcher found; no processes stopped.')


if __name__=='__main__':stop()
