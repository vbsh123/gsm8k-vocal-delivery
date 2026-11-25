#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode=${1:-start}
if (($#)); then shift; fi
unset PYTHONHOME PYTHONPATH TRANSFORMERS_CACHE
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
python_bin=${DIFFA_PYTHON:-.venv-diffa/bin/python}
log=outputs/diffa-cross-prompt94.log
case "$mode" in
    start)
        "$python_bin" -u collect_diffa_cross_prompt.py prepare "$@"
        nohup "$python_bin" -u collect_diffa_cross_prompt.py run "$@" >> "$log" 2>&1 < /dev/null &
        echo "Started PID $! · $log"
        ;;
    run|prepare|summary)
        exec "$python_bin" -u collect_diffa_cross_prompt.py "$mode" "$@"
        ;;
    *)
        echo "Usage: bash cross_prompt94.sh {start|run|prepare|summary} [--audio-root DIR]" >&2
        exit 2
        ;;
esac
