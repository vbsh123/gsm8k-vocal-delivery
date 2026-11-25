#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

out=outputs/mai-olivia-angry-gsm100
mode=${1:-run}
case "$mode" in
  score) exec .venv-emotion/bin/python -u classify_cloud_smoke.py --device cpu --path "$out" ;;
  run|prepare) ;;
  *) echo "Usage: bash run_mai_olivia_angry100.sh [run|prepare|score]" >&2; exit 2 ;;
esac

mkdir -p "$out"
python3 - "$out/questions.jsonl" <<'PY'
import json
import random
import sys
from pathlib import Path

bank = json.loads(Path("kimi_gsm8k_questions.json").read_text())
questions = random.Random(42).sample(bank["questions"], 100)
assert len({row["question_id"] for row in questions}) == 100
Path(sys.argv[1]).write_text("".join(json.dumps(row) + "\n" for row in questions))
PY

options=()
if [[ $mode == prepare ]]; then options+=(--prepare-only); fi
bash run_mai_gsm_smoke.sh --input-jsonl "$out/questions.jsonl" --num-examples 100 \
  --styles angry --output "$out" "${options[@]}"
if [[ $mode == run ]]; then
  exec .venv-emotion/bin/python -u classify_cloud_smoke.py --device cpu --path "$out"
fi
