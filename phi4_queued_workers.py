"""GPU admission and child-process cleanup for queued Phi worker groups."""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import signal
import subprocess
import time


@contextmanager
def gpu_queue():
    Path('outputs').mkdir(exist_ok=True)
    with Path('outputs/jenny-task-vectors.lock').open('a') as lock:
        print('Waiting for shared GPU lock.',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            print('Waiting for active GPU work to finish.',flush=True);time.sleep(30)
        yield


def run_workers(commands,output,ready_paths=None,release=None,on_ready=None):
    children=[];logs=[]
    previous=signal.getsignal(signal.SIGTERM)
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    try:
        for i,command in enumerate(commands):
            log=(output/f'worker{i}.log').open('a');logs.append(log)
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);children.append(child)
            print(f'STARTED worker{i} PID{child.pid}; log {output/f"worker{i}.log"}',flush=True)
        pending=on_ready
        while True:
            codes=[child.poll() for child in children]
            failures=[(i,c) for i,c in enumerate(codes) if c is not None and c!=0]
            if failures:raise RuntimeError(f'Phi workers failed: {failures}; inspect worker logs. Saved work retained.')
            if pending is not None and all(path.exists() for path in ready_paths):
                pending();release.write_text('ready\n');pending=None
            if all(code is not None for code in codes):
                if pending is not None:raise RuntimeError('Collection workers exited without the merge barrier.')
                break
            time.sleep(1)
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:child.wait()
        for log in logs:log.close()
        signal.signal(signal.SIGTERM,previous)
