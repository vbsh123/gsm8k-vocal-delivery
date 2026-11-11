#!/usr/bin/env bash
# Current dataset audit -> Jenny emotion labels -> Qwen -> Kimi, one GPU worker at a time.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
exec 5>outputs/azure-after-classification.lock
flock 5
export PIP_NO_CACHE_DIR=1
conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$conda_exe" ]]; then
  echo 'Conda is missing; use the machine with the existing Qwen/Kimi environments.' >&2
  exit 1
fi
export PATH="$(dirname "$conda_exe"):$PATH"

python - <<'PY'
from pathlib import Path
import json
import os
import shutil
import wave

needed_models={'qwen-omni':False,'kimi-audio':False}
for name in ['azure-jenny-next500','azure-jenny-remaining-test']:
    folder=Path('outputs')/name
    plan=folder/'plan.jsonl'
    if not plan.exists():
        raise SystemExit(f'Missing {plan}; extract the upload TAR into this repository first.')
    rows=[json.loads(line) for line in plan.read_text().splitlines() if line.strip()]
    missing=[r['audio_path'] for r in rows if r['provider']=='azure' and not (folder/r['audio_path']).is_file()]
    if missing:
        raise SystemExit(f'{name}: {len(missing)} audio files missing; examples {missing[:3]}')
    print(f'{name}: {len(rows)} planned clips; audio present.')
    eligible=set()
    for row in rows:
        if row['provider']=='azure':
            with wave.open(str(folder/row['audio_path']),'rb') as audio:
                if 0<audio.getnframes()/audio.getframerate()<=30:
                    eligible.add(row['audio_path'])
    for model in ['qwen-omni','kimi-audio']:
        path=folder/f'results-{model}.jsonl'
        saved={json.loads(line)['audio_path'] for line in path.read_text().splitlines() if line.strip()} if path.exists() else set()
        pending=len(eligible-saved)
        needed_models[model] |= bool(pending)
        print(f'  {model}: {len(eligible & saved)} eligible answers saved; {pending} pending.')

# Missing model snapshots may require substantial disk; stop before a long wait.
missing_models=0
for model,repo,revision,default in [
    ('qwen-omni','Qwen--Qwen2.5-Omni-7B','ae9e1690543ffd5c0221dc27f79834d0294cba00','outputs/hf-qwen-omni'),
    ('kimi-audio','moonshotai--Kimi-Audio-7B-Instruct','9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b','outputs/hf-kimi-audio')]:
    home=Path(os.environ.get('HF_HOME') or default)
    snapshot=home/'hub'/f'models--{repo}'/'snapshots'/revision
    indexes=list(snapshot.glob('*.safetensors.index.json'))
    if indexes:
        shards=set(json.loads(indexes[0].read_text())['weight_map'].values())
        cached=all((snapshot/name).is_file() for name in shards)
    else:
        cached=(snapshot/'model.safetensors').is_file()
    missing_models+=not cached and needed_models[model]
needed=(4+25*missing_models)*2**30
free=shutil.disk_usage('.').free
print(f'Disk free: {free/2**30:.1f} GiB; uncached model estimate plus reserve: {needed/2**30:.1f} GiB.')
if free<needed:
    raise SystemExit('Insufficient estimated disk for model downloads. Free space or enlarge storage before starting this queue.')
PY

qwen_ready=1
kimi_ready=1
if ! "$conda_exe" run -n qwen-omni python -c 'from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor; import librosa, soundfile' >/dev/null 2>&1; then
  echo "Setting up missing Qwen environment."
  bash setup_qwen_omni.sh || qwen_ready=0
fi
if ! "$conda_exe" run -n kimi-audio python -c 'import flash_attn; from kimia_infer.api.kimia import KimiAudio' >/dev/null 2>&1; then
  echo "Setting up missing Kimi environment."
  bash setup_kimi_audio.sh || kimi_ready=0
fi

python -u wait_for_experiments.py --kind classification

# The new upload has audio but may not yet have its Jenny classifier labels.
if [[ -x .venv-emotion/bin/python ]]; then
  python -u run_azure_followups.py --stage classify || echo 'Jenny classification failed; model accuracy runs will still proceed unfiltered.'
else
  echo 'Emotion environment absent; existing labels will be used where available.'
fi
failed=0
if (( qwen_ready )); then
  python -u run_azure_followups.py --stage qwen || failed=1
else
  echo 'Qwen setup failed; its evaluation could not start.'
  failed=1
fi
if (( kimi_ready )); then
  python -u run_azure_followups.py --stage kimi || failed=1
else
  echo 'Kimi setup failed; its evaluation could not start.'
  failed=1
fi
echo "Queue finished; failure status=$failed. Existing results were retained."
exit "$failed"
