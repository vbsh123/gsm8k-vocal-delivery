"""Stop only a Flamingo wrapper blocked in flock; leave GPU/model jobs alone."""
import os
from pathlib import Path
import signal


def command(pid):
    try: return [os.fsdecode(a) for a in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if a]
    except OSError: return []


def stop():
    wrappers = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        args = command(path.name)
        if any(Path(a).name == 'evaluate_gsm_candidates.py' for a in args) and 'flamingo' in args and 'evaluate' in args:
            raise RuntimeError('Flamingo evaluation is already running; no processes stopped.')
        if any(Path(a).name == 'run_flamingo_gsm400.sh' for a in args): wrappers.append(int(path.name))
    waiting = []
    for pid in wrappers:
        try: children = [int(p) for p in (Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split()]
        except OSError: continue
        if not children or any(not command(p) or Path(command(p)[0]).name != 'flock' for p in children):
            raise RuntimeError(f'Flamingo wrapper{pid} is not solely waiting in flock; no processes stopped.')
        waiting.append((pid, children))
    for pid, children in waiting:
        for target in [pid]+children:
            try: os.kill(target, signal.SIGTERM)
            except ProcessLookupError: pass
        print(f'Stopped waiting Flamingo wrapper{pid} and its flock waiter; model jobs untouched.')
    if not waiting: print('No waiting Flamingo wrapper found; no processes stopped.')


if __name__ == '__main__': stop()
