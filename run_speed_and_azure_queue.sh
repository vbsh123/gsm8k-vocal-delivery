#!/usr/bin/env bash
# DIFFA speed follow-up, then the two newer Azure batches on Qwen and Kimi.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 7>outputs/speed-and-azure-queue.lock
flock 7

# Check all inputs before starting a long job.
.venv-diffa/bin/python - <<'PY'
from pathlib import Path
import json
from importlib.metadata import version

assert version('audiostretchy') == '1.3.14', 'Install audiostretchy==1.3.14 in .venv-diffa'
pilot=Path('outputs/spoken-gsm-audiostretchy200/run_config.json')
if not pilot.is_file():
    raise SystemExit(f'Missing {pilot}: copy it from the original speed run to exclude its 200 questions.')
missing=[]
for folder in [Path('outputs/azure-jenny-next500'),Path('outputs/azure-jenny-remaining-test')]:
    plan=folder/'plan.jsonl'
    if not plan.exists():
        missing.append(str(plan))
        continue
    for line in plan.read_text().splitlines():
        if line.strip():
            row=json.loads(line)
            path=folder/row['audio_path']
            if row['provider']=='azure' and not path.is_file():
                missing.append(str(path))
if missing:
    raise SystemExit(f'{len(missing)} input files missing. Copy the two Azure folders before starting. Examples: {missing[:5]}')
print('Input plans and audio are present. Saved results are skipped only if their JSONL files are present too.')
PY
for env_name in qwen-omni kimi-audio; do
  conda run -n "$env_name" python -c 'import sys; print(sys.executable)'
done

echo "Stage 1/3: DIFFA, remaining Spoken GSM at 1x/1.2x/1.3x."
bash run_audiostretchy_speed_remaining.sh
echo "Stage 2/3: Qwen, Azure emotional-delivery batches."
python -u run_azure_followups.py --stage qwen
echo "Stage 3/3: Kimi, Azure emotional-delivery batches."
python -u run_azure_followups.py --stage kimi
echo "Speed and Azure evaluation queue finished."
