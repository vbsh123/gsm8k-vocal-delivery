#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

out=outputs/jenny-terrified-gsm30
case "${1:-run}" in
  run|prepare) ;;
  *) echo "Usage: bash run_jenny_terrified30.sh [run|prepare]" >&2; exit 2 ;;
esac

mkdir -p "$out"
python3 - "$out/questions.jsonl" <<'PY'
import json
import sys
from pathlib import Path

bank = json.loads(Path("data/spoken_prompt_math50.json").read_text())
questions = bank["datasets"]["gsm8k"]["questions"][2:32]
assert len({row["question_id"] for row in questions}) == 30
Path(sys.argv[1]).write_text("".join(json.dumps(row) + "\n" for row in questions))
PY

export AZURE_SPEECH_KEY="${AZURE_SPEECH_KEY:-${SPEECH_KEY:-}}"
export AZURE_SPEECH_REGION="${AZURE_SPEECH_REGION:-${SPEECH_REGION:-}}"
options=()
if [[ ${1:-run} == prepare ]]; then
  options+=(--prepare-only)
fi
python3 cloud_tts_smoke.py --provider azure --input-jsonl "$out/questions.jsonl" \
  --num-examples 30 --azure-voice en-US-JennyNeural --azure-intensity 1.5 \
  --styles terrified --output "$out" --resume "${options[@]}"
if [[ ${1:-run} == prepare ]]; then exit 0; fi

.venv-emotion/bin/python -u classify_cloud_smoke.py --device cpu --path "$out"
python3 - "$out" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

folder = Path(sys.argv[1])
def read_rows(name):
    return [json.loads(line) for line in (folder / name).read_text().splitlines() if line.strip()]

expected = {row["audio_path"] for row in read_rows("plan.jsonl")}
predicted = {row["audio_path"]: row["emotion"] for row in read_rows("results-emotion2vec.jsonl")}
if expected != predicted.keys() or len(expected) != 30:
    sys.exit("Incomplete E2V results. Rerun to resume.")
counts = Counter(predicted.values())
print(f"Jenny terrified classified as fearful: {counts['fearful']}/30 ({counts['fearful']/30:.1%})")
print("Predicted labels:", dict(counts.most_common()))
PY
