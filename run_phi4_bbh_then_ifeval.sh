#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
echo 'STAGE: Queue BBH3+BBH5 combined steering; details in outputs/phi4-bbh-combo.log'
bash run_phi4_bbh_combo.sh > outputs/phi4-bbh-combo.log 2>&1
echo 'STAGE: Queue three-worker IFEval all32 audio+answer sweep; details in outputs/phi4-ifeval-all-layers.log'
bash run_phi4_ifeval_all_layers.sh > outputs/phi4-ifeval-all-layers.log 2>&1
echo 'Finished BBH combined steering and IFEval all-layer sweep.'
