#!/usr/bin/env bash
# Separate environment; use the upstream installation recipe and pinned source.
set -euo pipefail
export PIP_NO_CACHE_DIR=1
cd "$(dirname "$0")"
command -v conda >/dev/null || { echo 'Conda is required for the upstream Python 3.10/audio setup.'; exit 1; }
revision=d6e3ca874f399a976c42bd3c8a58eb469eb7e3a3
source_dir="$PWD/external/Dynin-Omni"
if [[ ! -d "$source_dir" ]]; then
    git clone https://github.com/AIDASLab/Dynin-Omni.git "$source_dir"
fi
git -C "$source_dir" checkout "$revision"
if ! conda run -n dynin-omni python -c 'import sys; assert sys.version_info[:2] == (3, 10)' >/dev/null 2>&1; then
    conda create -y -n dynin-omni python=3.10
fi
conda run --no-capture-output -n dynin-omni bash "$source_dir/scripts/init_env.sh"
echo 'Ready: conda run --no-capture-output -n dynin-omni python -u evaluate_dynin_omni.py'
