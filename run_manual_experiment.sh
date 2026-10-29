#!/usr/bin/env bash
# Prepare wording, generate speech, then crop; stop before filtering or evaluation.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

example_count="${1:-200}"
experiment_dir="${2:-outputs/manual200-mixed}"
source_files=("${@:3}")
if [[ ${#source_files[@]} -eq 0 ]]; then
  source_files=(outputs/pilot50-v2/questions.jsonl outputs/extra150-v2/questions.jsonl)
fi

export PYTHONUNBUFFERED=1
mkdir -p "$experiment_dir"
# Keep the saved question selection when resuming an interrupted run.
if [[ ! -f "$experiment_dir/source.jsonl" ]]; then
  .venv/bin/python filter_questions.py \
    --sources "${source_files[@]}" --num-examples "$example_count" --add-new \
    --output "$experiment_dir/source.jsonl"
fi
saved_count="$(.venv/bin/python -c \
  'import sys; print(sum(bool(line.strip()) for line in open(sys.argv[1])))' \
  "$experiment_dir/source.jsonl")"
if (( example_count > saved_count )); then
  .venv/bin/python filter_questions.py \
    --sources "${source_files[@]}" --existing "$experiment_dir/source.jsonl" \
    --num-examples "$example_count" --add-new --output "$experiment_dir/source.jsonl"
fi
example_count="$(.venv/bin/python -c \
  'import sys; print(sum(bool(line.strip()) for line in open(sys.argv[1])))' \
  "$experiment_dir/source.jsonl")"
printf 'Selected %s questions total.\n' "$example_count"

.venv/bin/python prepare_text.py \
  --input-jsonl "$experiment_dir/source.jsonl" \
  --style all --num-examples "$example_count" --output "$experiment_dir/text" \
  | tee "$experiment_dir/preparation.log"

.venv/bin/python generate.py \
  --input-jsonl "$experiment_dir/text/tts-input.jsonl" \
  --num-examples "$example_count" --output "$experiment_dir/recordings" --extend \
  | tee -a "$experiment_dir/generation.log"

.venv-diffa/bin/python crop_questions.py \
  --path "$experiment_dir/recordings" --output "$experiment_dir/cropped" \
  --device cuda | tee -a "$experiment_dir/cropping.log"

printf 'Finished. Cropped recordings: %s/cropped. Filtering and evaluation have not run.\n' "$experiment_dir"
