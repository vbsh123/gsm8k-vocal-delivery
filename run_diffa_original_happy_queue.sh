#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Preserve the exact original angry collection IDs, original prompts and held-out IDs.
exec bash run_diffa_happy_nightly.sh --datasets gsm8k svamp multiarith bbh3 bbh5 ifeval --stages primary --strengths 1 --require-existing-audio "$@"
