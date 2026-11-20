#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -gt 0 ]]; then
    exec bash run_phi4_gsm_combo.sh --dataset svamp "$@"
fi
# Complete matching500/500 standalone controls and vectors before combining.
# Existing compatible runs resume without regenerating completed work.
bash run_phi4_sweep.sh --datasets svamp --layers 0 --group audio --strengths -1 --no-wait
bash run_phi4_sweep.sh --datasets svamp --layers 30 --group answer --strengths -2 --no-wait
exec bash run_phi4_gsm_combo.sh --dataset svamp
