#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 9>outputs/multi-extra-happy-parallel-wrapper.lock
flock -n 9 || { echo 'MultiArith/happy follow-up queue is already running.'; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
echo 'STAGE: Prepare/reuse400 neutral/happy clips; GPU jobs continue.'
.venv-diffa/bin/python -u evaluate_gsm_happy400.py --mode audio
echo 'STAGE: Wait for earlier Phi/MiniCPM and happy queues; no jobs stopped.'
for file in gsm-happy400-wrapper phi-minicpm-gsm-parallel-wrapper minicpm-gsm400-steering-wrapper \
  phi4-reverse-followups-wrapper phi4-gsm-extra-L30-a-2 phi4-gsm-extra-L30-a-1.5; do
    flock "outputs/$file.lock" true
done
# Check the new requested strengths/sources before starting either GPU job.
.venv-diffa/bin/python -u - <<'PY'
from pathlib import Path
from phi4_reverse_followups import prepare_multi_extra, folders, install_fixed_vector
out = Path('outputs/phi4-multiarith-reverse290-290-L17-extra-minus1.5-minus2')
out.mkdir(exist_ok=True)
install_fixed_vector(prepare_multi_extra(folders(17, 'multiarith'), out, 17), out)
PY
bash run_phi_multiarith_extra_then_happy.sh &
phi_job=$!
echo 'Waiting for Phi GPU initialization; MiniCPM happy400 will then join it.'
while kill -0 "$phi_job" 2>/dev/null; do
    if python - <<'PY'
from pathlib import Path
import subprocess
for pid in subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).splitlines():
    try: args = (Path('/proc')/pid.strip()/'cmdline').read_bytes().split(b'\0')
    except OSError: continue
    if any(a.endswith(b'phi4_reverse_followups.py') for a in args) and b'multiarith-extra' in args:
        if Path('outputs/phi4-multiarith-reverse290-290-L17-extra-minus1.5-minus2/runtime.json').exists(): raise SystemExit(0)
    if any(a.endswith(b'evaluate_gsm_happy400.py') for a in args) and b'phi4' in args:
        raise SystemExit(0)
raise SystemExit(1)
PY
    then break; fi
    sleep 15
done
mini_status=0
.venv-bbh-minicpm/bin/python -u evaluate_gsm_happy400.py --mode evaluate --model minicpm --concurrent \
  >outputs/minicpm-gsm-happy400.log 2>&1 || mini_status=$?
if [[ "$mini_status" == 0 ]]; then
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u evaluate_gsm_happy400.py --mode summary --model minicpm \
      >>outputs/minicpm-gsm-happy400.log 2>&1 || mini_status=$?
fi
phi_status=0
wait "$phi_job" || phi_status=$?
echo "Finished: Phi exit=$phi_status; MiniCPM exit=$mini_status."
(( phi_status == 0 && mini_status == 0 ))
