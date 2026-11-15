#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${DIFFA_PYTHON:-.venv-diffa/bin/python}"
"$PYTHON" -m pip install --no-cache-dir -c requirements-diffa.txt -r requirements-harness.txt
"$PYTHON" - <<'PY'
import importlib.metadata
import json
from pathlib import Path
from rescore_audio_steering import HarnessTaskScorer
from score_bbh_harness import HarnessScorer
metadata = importlib.metadata.distribution('lm_eval')
source = json.loads(metadata.read_text('direct_url.json') or '{}')
assert source.get('vcs_info', {}).get('commit_id') == 'd6de81643928d653435c431bae19945d41d32520'
math = HarnessTaskScorer('gsm8k', Path('outputs/harness-api-setup/gsm8k'))
assert all(v[0]['correct'] for v in math.score(['#### 5'], [dict(question='2+3?', answer='5')]).values())
bbh = HarnessScorer(Path('outputs/harness-api-setup/bbh'))
doc = dict(input='Which?\nOptions:\n(A) red\n(B) blue\n(C) green', target='(B)')
assert all(v[0]['correct'] for v in bbh.score(['The answer is (B).'], [doc]).values())
print('Installed harness task API verified; no model was loaded.')
PY
