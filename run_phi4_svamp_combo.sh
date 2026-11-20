#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Reuse existing own-task vectors and neutral answers; generate combined answers only.
exec bash run_phi4_gsm_combo.sh --dataset svamp "$@"
