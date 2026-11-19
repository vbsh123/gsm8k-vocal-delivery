"""Stop this repo's nightly wrapper/opposite diagnostic; preserve completed files and other jobs."""
import os
from pathlib import Path
import signal
import time


def targets(root,proc=Path('/proc')):
    found=[]
    for path in proc.iterdir():
        if not path.name.isdigit() or int(path.name)==os.getpid():continue
        try:
            if (path/'cwd').resolve()!=root.resolve():continue
            args=[os.fsdecode(a) for a in (path/'cmdline').read_bytes().split(b'\0') if a]
            names={Path(a).name for a in args}
            wrapper='run_phi4_nightly_followups.sh' in names
            opposite=('phi4_opposite_recovery.py' in names and '--mode' in args
                and args[args.index('--mode')+1:args.index('--mode')+2]==['run'])
            if wrapper or opposite:found.append((int(path.name),wrapper))
        except OSError:continue
    return sorted(found,key=lambda item:not item[1])


def stop():
    selected=targets(Path.cwd())
    # Pause parents so they cannot advance to another experiment when the child exits.
    for pid,wrapper in selected:
        if wrapper:
            try:os.kill(pid,signal.SIGSTOP)
            except ProcessLookupError:pass
    for pid,wrapper in selected:
        try:
            os.kill(pid,signal.SIGTERM)
            if wrapper:os.kill(pid,signal.SIGCONT)
        except ProcessLookupError:pass
        print(f'Sent stop to {"nightly wrapper" if wrapper else "opposite diagnostic"} PID{pid}.',flush=True)
    for _ in range(50):
        if not targets(Path.cwd()):break
        time.sleep(.2)
    else:raise RuntimeError('A matched process is still exiting; wait/check ps before restarting. No files removed.')
    print('Completed answers retained. No other experiments stopped.')


if __name__=='__main__':stop()
