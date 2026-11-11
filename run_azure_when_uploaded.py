"""Wait for the verified audio upload and classification queue, then run Jenny/Qwen/Kimi."""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from wait_for_experiments import active_jobs

EXPECTED_BYTES=3585628160
EXPECTED_SHA256='c35b495dccaa1374cf7066a4f4c78ea267aea82a0e2f963390760934afbcd333'


def fingerprint(path):
    stat=path.stat()
    return stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino


def archive_complete(path, expected_bytes, expected_sha256):
    try:
        before=fingerprint(path)
        if before[0]!=expected_bytes:
            return False
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):
                digest.update(chunk)
        return digest.hexdigest()==expected_sha256 and before==fingerprint(path)
    except OSError:
        return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=Path('/workspace/azure-new-batches-upload.tar'))
    p.add_argument('--poll-seconds',type=float,default=30)
    args=p.parse_args()
    if args.poll_seconds<=0:
        p.error('--poll-seconds must be positive')
    root=Path(__file__).resolve().parent
    os.chdir(root)
    Path('outputs').mkdir(exist_ok=True)
    previous=verified=last_checked=None
    with Path('outputs/upload-and-models.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        while True:
            try:
                current=fingerprint(args.archive)
            except OSError:
                current=None
            size=current[0] if current else 0
            if current!=verified:
                verified=None
            if current and size==EXPECTED_BYTES and current==previous and current!=last_checked:
                print('Exact upload size reached; checking SHA-256.',flush=True)
                if archive_complete(args.archive,EXPECTED_BYTES,EXPECTED_SHA256):
                    verified=current
                else:
                    print('Checksum not ready/matching; waiting for the upload to finish.',flush=True)
                last_checked=current
            jobs=active_jobs('classification',root)
            print(f'Upload: {size/1e6:.1f}/{EXPECTED_BYTES/1e6:.1f} MB; checksum verified={verified is not None}; classification PID(s)={jobs}',flush=True)
            if verified is not None and not jobs:
                with tarfile.open(args.archive,'r:') as archive:
                    missing_bytes=sum(m.size for m in archive if m.isfile() and not (root/m.name).exists())
                if shutil.disk_usage(root).free>=missing_bytes+512*1024*1024:
                    break
                print(f'Waiting for extraction space: need about {missing_bytes/2**30:.1f} GiB plus reserve.',flush=True)
            previous=current
            time.sleep(args.poll_seconds)
        print('Upload verified and classification processes exited. Extracting without overwriting existing files.',flush=True)
        subprocess.run(['tar','--skip-old-files','-xf',str(args.archive),'-C',str(root)],check=True)
        subprocess.run(['bash','run_azure_after_classification.sh'],check=True)


if __name__=='__main__':
    main()
