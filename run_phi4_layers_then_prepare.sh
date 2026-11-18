#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo 'STAGE: Two Phi diagnostic workers; no shared GPU queue or job dependency.'
bash run_phi4_all_layers_split.sh test
echo 'STAGE: Prepare SVAMP and BBH3/5 audio and frozen splits; CPU/TTS only.'
bash prepare_phi4_next_tasks.sh
echo 'Finished. Inspect the layer diagnostic, then choose layer/group/strength for task steering.'
