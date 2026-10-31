#!/usr/bin/env bash
# Extend the saved selection, prepare style wording, and generate speech.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

example_count="${1:-500}"
experiment_dir="${2:-outputs/manual200-mixed}"
export PYTHONUNBUFFERED=1

.venv/bin/python filter_questions.py \
  --sources outputs/pilot50-v2/questions.jsonl outputs/extra150-v2/questions.jsonl \
  --existing "$experiment_dir/source.jsonl" \
  --num-examples "$example_count" --add-new \
  --output "$experiment_dir/source.jsonl"

.venv/bin/python prepare_text.py \
  --input-jsonl "$experiment_dir/source.jsonl" \
  --style all --num-examples "$example_count" \
  --output "$experiment_dir/text"

.venv/bin/python generate.py \
  --input-jsonl "$experiment_dir/text/tts-input.jsonl" \
  --num-examples "$example_count" \
  --output "$experiment_dir/recordings" --extend
