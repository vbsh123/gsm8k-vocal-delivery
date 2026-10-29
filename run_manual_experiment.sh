#!/usr/bin/env bash
# Prepare wording, generate speech, then evaluate; each model exits before the next loads.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

example_count="${1:-200}"
experiment_dir="${2:-outputs/manual130}"
source_files=("${@:3}")
if [[ ${#source_files[@]} -eq 0 ]]; then
  source_files=(outputs/pilot50-v2/questions.jsonl outputs/extra150-v2/questions.jsonl)
fi

export PYTHONUNBUFFERED=1
mkdir -p "$experiment_dir"
example_count="$(.venv/bin/python filter_questions.py \
  --sources "${source_files[@]}" --num-examples "$example_count" \
  --output "$experiment_dir/source.jsonl")"
printf 'Selected %s questions whose previous recordings all fit 30 seconds.\n' "$example_count"

.venv/bin/python prepare_text.py \
  --input-jsonl "$experiment_dir/source.jsonl" \
  --style all --num-examples "$example_count" --output "$experiment_dir/text" \
  | tee "$experiment_dir/preparation.log"

.venv/bin/python generate.py \
  --input-jsonl "$experiment_dir/text/tts-input.jsonl" \
  --num-examples "$example_count" --output "$experiment_dir/recordings" \
  | tee "$experiment_dir/generation.log"

.venv-diffa/bin/python evaluate.py --path "$experiment_dir/recordings" \
  --max-new-tokens 512 --results-file results-reasoning-v2.jsonl \
  | tee "$experiment_dir/evaluation.log"
