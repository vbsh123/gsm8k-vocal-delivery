"""Stop this repo's nightly wrapper/opposite diagnostic; preserve completed files and other jobs."""
import argparse
import os
from pathlib import Path
import signal
import time


def targets(root,proc=Path('/proc'),include_l17_tasks=False,l7_only=False):
    found=[]
    for path in proc.iterdir():
        if not path.name.isdigit() or int(path.name)==os.getpid():continue
        try:
            if (path/'cwd').resolve()!=root.resolve():continue
            args=[os.fsdecode(a) for a in (path/'cmdline').read_bytes().split(b'\0') if a]
            names={Path(a).name for a in args}
            if l7_only:
                wrapper=bool(names & {'run_phi4_gsm_L7.sh','run_phi4_gsm_L7_split.sh'})
                model='run_phi4_gsm_L7.py' in names and '--mode' in args and args[args.index('--mode')+1:args.index('--mode')+2] in (['run'],['worker'])
                if wrapper or model:found.append((int(path.name),wrapper))
                continue
            wrapper='run_phi4_nightly_followups.sh' in names
            opposite=('phi4_opposite_recovery.py' in names and '--mode' in args
                and args[args.index('--mode')+1:args.index('--mode')+2]==['run'])
            task=False
            if include_l17_tasks and '--layer' in args and '--strength' in args:
                settings=args[args.index('--layer')+1:args.index('--layer')+2]==['17'] and args[args.index('--strength')+1:args.index('--strength')+2]==['-1.5']
                task_wrapper='run_phi4_prepared_tasks.sh' in names
                task_model='phi4_task_steering.py' in names and '--mode' in args and args[args.index('--mode')+1:args.index('--mode')+2]==['run']
                task=settings and (task_wrapper or task_model)
                wrapper=wrapper or settings and task_wrapper
            if wrapper or opposite or task:found.append((int(path.name),wrapper))
        except OSError:continue
    return sorted(found,key=lambda item:not item[1])


def stop(include_l17_tasks=False,l7_only=False):
    selected=targets(Path.cwd(),include_l17_tasks=include_l17_tasks,l7_only=l7_only)
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
        print(f'Sent stop to {"queue wrapper" if wrapper else "model process"} PID{pid}.',flush=True)
    for _ in range(50):
        if not targets(Path.cwd(),include_l17_tasks=include_l17_tasks,l7_only=l7_only):break
        time.sleep(.2)
    else:raise RuntimeError('A matched process is still exiting; wait/check ps before restarting. No files removed.')
    print('Completed answers retained. Only the selected L7 or nightly/opposite/L17 job scope was targeted.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-l17-tasks',action='store_true',help='Also stop the superseded L17 -1.5 task runner/model; retain completed answers.')
    parser.add_argument('--l7-only',action='store_true',help='Stop only L7 GSM wrappers/native workers to switch serial/parallel execution.')
    args=parser.parse_args()
    if args.l7_only and args.include_l17_tasks:parser.error('Choose either L7-only or L17 task cancellation.')
    stop(args.include_l17_tasks,args.l7_only)
