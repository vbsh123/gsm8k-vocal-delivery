# BBH three-object delivery evaluation: MOSS then Kimi

This is evaluation only. There is no activation collection or steering.
The queue runs MOSS-Audio-8B-Instruct first, then Kimi-Audio-7B-Instruct,
each in its own process/environment. It uses the existing Azure Jenny neutral
and angry recordings and the same BBH prompt used for DIFFA, including
`The answer is (<letter>).` No transcript, answer or intended emotion is supplied
in the model prompt. The model receives the real waveform and the fixed task
instruction. Both models use seed 42 and a 512-token generation budget.

MOSS uses greedy generation with the authors' processor, time markers and SDPA.
Kimi uses the same pinned greedy text-generation settings as our earlier Kimi
evaluations. Full recordings are passed to the upstream audio loaders/encoders.
Neither model uses our custom DIFFA chunking or a 30-second eligibility filter.
Kimi's supplied Whisper encoder itself chunks and joins long audio; MOSS has
its own chunked audio encoder. Original audio files are never modified.

## Run

To regenerate the 250 neutral/angry pairs on the evaluation machine rather than
transfer audio, export Azure credentials and use:

```bash
git pull --ff-only
read -rsp 'Azure Speech key: ' AZURE_SPEECH_KEY
echo
export AZURE_SPEECH_KEY
export AZURE_SPEECH_REGION=eastus
nohup bash run_bbh_moss_then_kimi.sh --regenerate-audio \
  > outputs/bbh-moss-then-kimi.log 2>&1 < /dev/null &
tail -f outputs/bbh-moss-then-kimi.log
```

This generates 500 clips with the same official questions, voice, intensity and
spoken wording, then evaluates MOSS and Kimi on those shared new recordings.
These are newly synthesized waveforms, not byte-identical copies of DIFFA's
earlier audio. Generation resumes missing clips and runs before the GPU wait.
Audio is saved in `outputs/azure-bbh-three-models-regenerated`; results are in
`outputs/bbh-three-regenerated-models/{moss,kimi}`. Restart with the same flag.
There is no archive/download/upload step.

```bash
.venv-diffa/bin/python evaluate_bbh_audio_models.py --model moss --mode summary --output outputs/bbh-three-regenerated-models
.venv-diffa/bin/python evaluate_bbh_audio_models.py --model kimi --mode summary --output outputs/bbh-three-regenerated-models
```

Alternatively, to reuse already-existing recordings:

Use a machine holding `outputs/azure-bbh-logical250-steering`, including the audio
files targeted by any symlinks, and the existing `.venv-diffa` harness environment.
If only `outputs/azure-bbh-logical200` exists, the default selects those 200 pairs
instead. An explicit `--source` overrides discovery. No Azure credentials or new
TTS generation is needed. Missing input files fail preflight before setup.

The script waits for current reasoning, mean-steering, IFEval/IFBench,
audio-model, speed/pitch and classification processes on this machine. Each model
also takes the shared GPU lock. It cannot observe jobs on another machine.

```bash
git pull --ff-only
df -h /workspace
nohup bash run_bbh_moss_then_kimi.sh \
  > outputs/bbh-moss-then-kimi.log 2>&1 < /dev/null &
tail -f outputs/bbh-moss-then-kimi.log
```

To specify a moved audio directory:

```bash
nohup bash run_bbh_moss_then_kimi.sh --source /path/to/azure-bbh-logical250-steering \
  > outputs/bbh-moss-then-kimi.log 2>&1 < /dev/null &
```

MOSS setup uses a separate Conda prefix `.venv-moss` (Python 3.12, NumPy 2,
upstream-pinned Torch 2.9.1/CUDA 12.8 and Transformers 4.57.1). It installs FFmpeg 7
and verifies the upstream loader preserves a 40-second waveform after resampling
before any MOSS weights are loaded. DIFFA and Kimi environments are unchanged.
Kimi reuses its existing `kimi-audio` environment, or calls the existing setup
script if needed. The scoring process runs through `.venv-diffa`, so the new
model environments do not need another harness installation.

Allow roughly 25–30 GiB additional disk for an uncached MOSS environment/weights;
uncached Kimi weights/encoders and a new Kimi environment need additional space.
The queue does not delete other models, recordings or results to make room.

## Results and resume

Raw responses, waveform hashes and pinned configurations are saved separately:

- `outputs/bbh-three-audio-models/moss/`
- `outputs/bbh-three-audio-models/kimi/`

Each recording resumes independently. For an optional four-generation smoke,
pass `--max-generations 4`; rerun without it to finish the same frozen cohort.

CPU-only results, including while a model is still running:

```bash
.venv-diffa/bin/python evaluate_bbh_audio_models.py --model moss --mode summary
.venv-diffa/bin/python evaluate_bbh_audio_models.py --model kimi --mode summary
```

The installed, pinned EleutherAI BBH task scores the unmodified responses with
strict and flexible filters. Tables use completed same-question/voice pairs,
with exact paired McNemar tests and 95% paired bootstrap confidence intervals.
Flexible extraction is primary; tests and intervals are unadjusted. The full
250-question table and original 125-question test-half table are both reported
when all 250 recordings are available. They overlap and are not independent
cohorts. This is an audio adaptation scored by upstream APIs, not the original
text-harness inference protocol.

Official implementations:

- MOSS source: `66326e6e0db34f036c86a76ba005efa4830c69dd`
- MOSS weights: `d4dc5a6d8cd79b43dcd82884c75e303b1ecd016d`
- [MOSS inference](https://github.com/OpenMOSS/MOSS-Audio/blob/66326e6e0db34f036c86a76ba005efa4830c69dd/infer.py)
- [Kimi long-audio Whisper encoder](https://github.com/MoonshotAI/Kimi-Audio/blob/349251e1d8f4f98d58fda59246381faecd7392e0/kimia_infer/models/tokenizer/whisper_Lv3/whisper.py)

Kimi code, model and tokenizer revisions remain those in `evaluate_kimi_audio.py`.

## Transfer the existing BBH audio to another machine

On the steering machine, package the source directory with symlinks dereferenced
so that recordings reused from the older 200-question directory become real
files in the archive:

```bash
tar --dereference -czf outputs/bbh-three-audio.tar.gz \
  -C outputs azure-bbh-logical250-steering
```

Download `outputs/bbh-three-audio.tar.gz` and upload it to `/workspace` on the
other machine. Extract into a separate location, then run the same queue there:

```bash
cd /workspace/gsm8k-vocal-delivery
git pull --ff-only
mkdir -p outputs/bbh-audio-transfer
tar -xzf /workspace/bbh-three-audio.tar.gz -C outputs/bbh-audio-transfer
nohup bash run_bbh_moss_then_kimi.sh \
  --source outputs/bbh-audio-transfer/azure-bbh-logical250-steering \
  > outputs/bbh-moss-then-kimi.log 2>&1 < /dev/null &
tail -f outputs/bbh-moss-then-kimi.log
```
