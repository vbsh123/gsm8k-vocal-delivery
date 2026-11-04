#!/usr/bin/env bash
# Wait for the audio-prefix run, then run the remaining experiments sequentially.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

prefix_pid="${1:?Usage: bash run_after_audio_prefix.sh AUDIO_PROCESS_PID}"
if [[ ! "$prefix_pid" =~ ^[0-9]+$ || "$prefix_pid" -le 1 ]]; then
  printf 'Pass the Python PID of the running audio-prefix experiment.\n' >&2
  exit 1
fi

process_args="$(ps -p "$prefix_pid" -o args= || true)"
if [[ -n "$process_args" && "$process_args" != *measure-audio-after-prompt-prefix.py* ]]; then
  printf 'PID %s belongs to another command: %s\n' "$prefix_pid" "$process_args" >&2
  exit 1
fi
printf '%s Waiting for audio-prefix PID %s to exit.\n' "$(date -Is)" "$prefix_pid"
while kill -0 "$prefix_pid" 2>/dev/null; do
  process_state="$(ps -p "$prefix_pid" -o stat= || true)"
  [[ -z "$process_state" || "$process_state" == Z* ]] && break
  sleep 15
done

# A process exiting does not tell us it finished successfully. Check saved results.
.venv-diffa/bin/python - <<'PY'
import json
from pathlib import Path

folder = Path("outputs/spoken-gsm-original")
conditions = {"baseline", "angry", "sad", "happy", "stressed", "afraid", "confident"}
with (folder / "manifest.jsonl").open() as stream:
    ids = [json.loads(line)["clip_id"] for line in stream if line.strip()][:300]
seen = set()
with (folder / "results-prompt-prefix.jsonl").open() as stream:
    for line in stream:
        if line.strip():
            row = json.loads(line)
            seen.add((row["clip_id"], row["condition"]))
missing = sum((clip, condition) not in seen for clip in ids for condition in conditions)
states = folder / "results-prompt-prefix.hidden-states.pt"
if len(ids) != 300 or missing or not states.exists():
    raise SystemExit(f"Audio run incomplete: {len(ids)} questions, {missing} missing conditions, "
                     f"hidden-state file exists={states.exists()}. Follow-ups were not started.")
print("All 300 audio questions have seven saved conditions; starting follow-ups.", flush=True)
PY

printf '%s Starting text-only framing.\n' "$(date -Is)"
.venv-diffa/bin/python -u measure-text-after-prompt-prefix.py --num-examples 300

printf '%s Starting emotion2vec unknown-vector extraction.\n' "$(date -Is)"
.venv-diffa/bin/python -u extract_unknown_vectors.py --classifier emotion2vec

printf '%s Starting SenseVoice unknown-vector extraction.\n' "$(date -Is)"
.venv-diffa/bin/python -u extract_unknown_vectors.py --classifier sensevoice

printf '%s All follow-up experiments finished.\n' "$(date -Is)"
