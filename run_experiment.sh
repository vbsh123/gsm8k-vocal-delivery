#!/usr/bin/env bash
# Run from a provisioned Vast instance; TTS exits before DIFFA loads.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
example_count="${1:-50}"
experiment_dir="${2:-outputs/pilot50-v2}"
sample_seed="${3:-123}"

if [[ ! -x .venv/bin/python || ! -x .venv-diffa/bin/python ]]; then
  printf 'Run bash setup_vast.sh and bash setup_vast.sh --diffa first.\n' >&2
  exit 1
fi
mkdir -p "$(dirname -- "$experiment_dir")"
exclude_files=()
if [[ -f outputs/pilot10/questions.jsonl ]]; then
  exclude_files+=(outputs/pilot10/questions.jsonl)
fi
# Arguments after the seed are previous question manifests to exclude.
for previous in "${@:4}"; do
  if [[ ! -f "$previous" ]]; then
    printf 'Previous questions file does not exist: %s\n' "$previous" >&2
    exit 1
  fi
  exclude_files+=("$previous")
done
exclude_args=()
if [[ ${#exclude_files[@]} -gt 0 ]]; then
  exclude_args=(--exclude-questions "${exclude_files[@]}")
fi
export PYTHONUNBUFFERED=1
.venv/bin/python generate.py \
  --num-examples "$example_count" --seed "$sample_seed" \
  --styles neutral confident hesitant happy angry sad stressed \
  --speakers Ryan --repeats 1 --output "$experiment_dir" \
  "${exclude_args[@]}" | tee "${experiment_dir}.generation.log"

.venv-diffa/bin/python evaluate.py --path "$experiment_dir" \
  --max-new-tokens 512 --results-file results-reasoning-v2.jsonl \
  | tee "$experiment_dir/evaluation.log"
