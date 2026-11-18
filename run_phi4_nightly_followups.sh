#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 8>outputs/phi4-nightly-followups.lock
flock -n 8 || { echo 'Phi nightly follow-ups already queued.'; exit 1; }
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
echo 'STAGE: Wait for current all-layer diagnostic and its task-preparation wrapper; no jobs stopped.'
.venv-diffa/bin/python -u wait_for_experiments.py --kind phi-layer-nightly-inputs
# Confirm the existing diagnostic really finished, rather than proceeding after a failed wrapper.
.venv-diffa/bin/python - <<'PY'
import json
from pathlib import Path
import phi4_all_layer_diagnostic as d
out=Path('outputs/phi4-gsm400-all-layers-test869-minus1')
plan=json.loads((out/'run_config.json').read_text());challenge=json.loads((out/'challenge.json').read_text())
_,done=d.completed(plan,out,challenge)
expected=len(challenge['cases'])*len(plan['layers'])*len(plan['groups'])
if len(done)!=expected:raise RuntimeError(f'Finish current diagnostic first: {len(done)}/{expected}.')
print('Original diagnostic complete:',len(done),'answers.')
PY
echo 'STAGE: Wait for shared GPU lock and active GPU work; then run all new Phi tasks sequentially.'
exec 9>outputs/jenny-task-vectors.lock
flock 9
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
    echo 'Waiting for active GPU work.'
    sleep 30
done
echo 'STAGE: Opposite cohort, neutral correct / angry wrong. Reuse400 vectors; both input styles; all32 layers -1.'
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u phi4_opposite_recovery.py --mode prepare >outputs/phi4-opposite-recovery.log 2>&1
.venv-bbh-phi4/bin/python -u phi4_opposite_recovery.py --mode run >>outputs/phi4-opposite-recovery.log 2>&1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u phi4_opposite_recovery.py --mode summary >>outputs/phi4-opposite-recovery.log 2>&1
echo 'STAGE: Prepare/resume SVAMP500/500 and BBH3/5 125/125 audio and splits.'
bash prepare_phi4_next_tasks.sh >outputs/phi4-next-tasks-prepare.log 2>&1
echo 'STAGE: SVAMP, BBH3, BBH5 sequentially; own-task vectors; L17 answer -1.5; full held-out halves.'
bash run_phi4_prepared_tasks.sh --layer 17 --group answer --strength -1.5 >outputs/phi4-prepared-tasks-L17-minus1.5.log 2>&1
echo 'Finished Phi nightly opposite diagnostic and three task steering runs.'
