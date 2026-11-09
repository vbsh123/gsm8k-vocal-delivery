#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
.venv-diffa/bin/python -u extract_jenny_svd_probe.py --limit 2 --output outputs/jenny-svd-probe-smoke
.venv-diffa/bin/python -u extract_jenny_svd_probe.py
.venv-diffa/bin/python -u evaluate_jenny_svd_probe.py
