#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/phi-minicpm-gsm-parallel-wrapper.lock
flock -n 9 || { echo 'Parallel GSM queue is already running.'; exit 1; }
PHI_CONCURRENT=1 bash run_phi4_gsm_extra.sh 30 -2 >outputs/phi4-gsm-L30-minus2.log 2>&1 &
phi_job=$!
# Start Phi first so an older Phi queue never waits for the new MiniCPM
# process between its stages. Once Phi owns GPU memory, MiniCPM may join it.
echo 'Waiting for Phi L30 -2 to load on GPU, then starting MiniCPM alongside it.'
while kill -0 "$phi_job" 2>/dev/null; do
    if python - <<'PY'
from pathlib import Path
import subprocess
pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).splitlines()
for pid in pids:
    try: args = (Path('/proc')/pid.strip()/'cmdline').read_bytes().split(b'\0')
    except (OSError, ValueError): continue
    if any(a.endswith(b'phi4_reverse_followups.py') for a in args) and b'gsm-extra' in args and b'-2' in args:
        raise SystemExit(0 if Path('outputs/phi4-gsm-reverse400-869-L30-a-2/runtime.json').exists() else 1)
raise SystemExit(1)
PY
    then break; fi
    sleep 15
done
echo 'Starting MiniCPM L30 +1:400 collection pairs,869 tests.'
mini_status=0
bash run_minicpm_gsm_steering.sh >outputs/minicpm-gsm400-869-L30-a1.log 2>&1 || mini_status=$?
phi_status=0
wait "$phi_job" || phi_status=$?
echo "Finished: Phi exit=$phi_status; MiniCPM exit=$mini_status."
(( phi_status == 0 && mini_status == 0 ))
