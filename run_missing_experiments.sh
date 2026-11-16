#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export DIFFA_PYTHON="${DIFFA_PYTHON:-$PWD/.venv-diffa/bin/python}"
exec "$DIFFA_PYTHON" -u run_missing_experiments.py "$@"
