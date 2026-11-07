#!/usr/bin/env bash
# Extract on batch two, then test neutral/sad/happy from batch one.
# Saved audio only; sequential DIFFA processes; no Azure calls or credentials.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=.venv-diffa/bin/python
VECTOR_DIR=outputs/jenny-angry-vectors-second500
RESULT_DIR=outputs/steer-jenny-second-vector-first500
[[ -x "$PYTHON" ]] || { echo "Missing $PYTHON; run this on the DIFFA machine." >&2; exit 1; }

echo "$(date -Is) Checking saved audio, classifications, and question separation."
"$PYTHON" -u - <<'PY'
import json
from pathlib import Path
from extract_jenny_angry import recordings, select_pairs
from steer_jenny_deliveries import select_recordings

source = Path("outputs/azure-jenny-next500")
targets = [Path("outputs/azure-jenny500-neutral"), Path("outputs/azure-jenny500-extra")]
records, labels = recordings([source])
pairs = select_pairs(records, labels, seed=42)
selected, skipped = select_recordings(targets, ["neutral", "sad", "happy"])
if not pairs or not selected:
    raise SystemExit("No extraction pairs or test recordings; check saved audio and emotion2vec results.")
train_ids = {r["question_id"] for r in records.values()}
test_ids = {r["question_id"] for r in selected}
if train_ids & test_ids:
    raise SystemExit("Extraction and test questions overlap; check input batch folders.")
prior = json.loads(Path("outputs/steer-jenny-angry-L0-25-n50/run_config.json").read_text())
if prior["settings"]["seed"] != 42:
    raise SystemExit("Expected existing steering base seed 42; check the prior run configuration.")
from collections import Counter
print(f"Extraction: {len(pairs)} emotion2vec-agreed angry/neutral pairs, {2*len(pairs)} generations.")
print("Test clips by requested delivery:", dict(Counter(r["style"] for r in selected)))
print("Skipped invalid/over-30-second test clips:", dict(skipped))
print(f"Steering: {len(selected)*3} generations before resume (baseline + L11 answer x1 + L12 acoustic x1).")
print("Extraction/test questions are disjoint. This reverse-split comparison remains exploratory.")
PY

echo "$(date -Is) Extracting second-batch angry-minus-neutral directions."
"$PYTHON" -u extract_jenny_angry.py \
  --paths outputs/azure-jenny-next500 \
  --output "$VECTOR_DIR" --seed 42

echo "$(date -Is) Testing new vector on first-batch neutral, sad, and happy audio."
"$PYTHON" -u steer_jenny_deliveries.py \
  --paths outputs/azure-jenny500-neutral outputs/azure-jenny500-extra \
  --styles neutral sad happy \
  --vectors "$VECTOR_DIR/style-vectors.pt" \
  --output "$RESULT_DIR"

echo "$(date -Is) Second-batch vector extraction and first-batch steering finished."
