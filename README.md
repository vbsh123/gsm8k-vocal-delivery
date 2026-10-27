# GSM8K vocal delivery pilot

Generate the **same math questions** in seven requested vocal deliveries:
**neutral, confident, hesitant, happy, angry, sad, stressed**.

Uses original [GSM8K](https://huggingface.co/datasets/openai/gsm8k) question text and
[Qwen3-TTS-12Hz-1.7B-CustomVoice](https://github.com/QwenLM/Qwen3-TTS) to synthesize
new recordings. The TTS reads supplied questions; it does **not** invent problems or
solve them. Neutral audio is generated with the same model and speaker as every
other style. The answer and solution are saved for scoring but never passed to TTS.

This repo generates stimuli for a later DIFFA/audio-model experiment. It does not
run DIFFA, extract steering vectors, or claim that requested emotions were realized.

## Vast setup

An A40 with about 45 GB usable VRAM should have ample room for sequential BF16
inference with this 1.7B TTS model. Actual peak memory and speed have not been
measured by this project. Use a Linux CUDA instance with Python 3.12 recommended,
a driver compatible with CUDA 12.8, and roughly 25 GB or more free disk space for
the environment, Hugging Face caches, and recordings.

```bash
git clone https://github.com/vbsh123/gsm8k-vocal-delivery.git
cd gsm8k-vocal-delivery
bash setup_vast.sh
source .venv/bin/activate
```

The setup script creates `.venv`, installs CUDA 12.8 PyTorch/torchaudio 2.9.1 and
the Python requirements, and checks CUDA availability. It uses standard PyTorch
SDPA attention; FlashAttention is optional, not required.
On minimal Ubuntu images, install `python3-venv`, `sox`, and `libsndfile1` first
if missing (`apt-get update` then `apt-get install -y python3-venv sox libsndfile1`).
If your driver needs different PyTorch CUDA wheels, create the venv and install
matching `torch`/`torchaudio` wheels from [PyTorch](https://pytorch.org/get-started/locally/),
then `pip install -r requirements.txt` instead of running the setup script.

Run in `tmux` if your SSH session might disconnect. Store outputs and the Hugging
Face cache on persistent storage if you plan to destroy/recreate the instance.

## DIFFA-2 setup for evaluation

When the recordings are ready, install the same DIFFA-2 source revision and
dependency versions as the caching experiments:

```bash
bash setup_vast.sh --diffa
source .venv-diffa/bin/activate
```

`bash setup_diffa.sh` is equivalent. This requires **Python 3.10–3.12** and an
NVIDIA driver compatible with CUDA 12.4. If `python3` is a different version:

```bash
PYTHON_BIN=python3.12 bash setup_vast.sh --diffa
```

The script automatically clones the official implementation into `external/DIFFA`
at commit `0a87594e110984fbaf00ebf5bfe5840c212e4438`. It creates `.venv-diffa`,
installs PyTorch/torchaudio 2.6.0 and Transformers 4.49.0, and registers the
`DIFFA-2` source directory inside that environment. You can then import
`src.modeling_DIFFA` and `src.configuration_DIFFA` from your evaluation script
without another clone or a manual `PYTHONPATH` setting.

The TTS environment uses a different Transformers version, so use `.venv` for
generation and `.venv-diffa` for evaluation. Setup checks source imports and CUDA;
it does not load model weights or run an evaluation. Model weights download when
your loader requests them. The previous experiments used
`zhoujiaming777/DIFFA-2`, `GSAI-ML/LLaDA-8B-Instruct`, and `openai/whisper-large-v3`.
The DIFFA-1 / Whisper-small loading example is a different model configuration.

Run the evaluator with the prompt that requests brief reasoning:

```bash
python evaluate.py --path outputs/pilot10 --results-file results-reasoning.jsonl
```

The evaluator reads WAV headers before loading weights. If **any recording of a
question exceeds 30 seconds**, all recordings of that question are skipped, so
styles are compared on the same questions. Audio is neither chunked nor cropped.
Exactly 30 seconds is accepted. Skipped clip IDs and durations are written to
`results-reasoning.jsonl.skipped.jsonl`. If no questions qualify, the script exits
before downloading/loading model weights. Results include the exact prompt and
the raw response; the prompt requests reasoning but does not guarantee it.
Use a new results filename when changing the prompt: existing files are appended
to, and reruns do not automatically skip completed evaluations.

## Generate a pilot

Start with one question to check the model and listen to all seven styles:

```bash
python generate.py --num-examples 1 --output outputs/smoke
```

Then generate ten questions, seven styles, one voice: **70 WAVs**.

```bash
python generate.py --num-examples 10 --output outputs/pilot10
```

Two voices produce **140 WAVs**:

```bash
python generate.py --num-examples 10 --speakers Ryan Aiden --output outputs/pilot10-two-voices
```

Control count, styles, repeats, split, and seed:

```bash
python generate.py \
  --num-examples 30 \
  --split train \
  --styles neutral confident hesitant happy angry sad stressed \
  --speakers Ryan Aiden \
  --repeats 2 \
  --seed 123 \
  --max-words 120 \
  --output outputs/train30
```

Clip count = questions × styles × voices × repeats. Default split is `test`,
default seed is 42, default voice is Ryan, default repeats is 1.
`--max-words` optionally filters long questions **before** sampling; default is
no filtering. The question shuffle is deterministic, and increasing the count
with the same seed/filter extends the same question sequence. Split by question
ID before using recordings for vector extraction, tuning, and held-out evaluation.

The styles are editable in [configs/styles.json](configs/styles.json). Use
`--styles-file path/to/custom.json` for another mapping. Each style instruction
gets the same suffix asking TTS to preserve the supplied words and numbers.
"Stressed" means anxious/pressured delivery, not answer-informed word emphasis.
The spelling `hesistant` is accepted as an alias for `hesitant`.

### Plan without loading the model

```bash
python generate.py --num-examples 10 --output outputs/pilot10 --prepare-only
```

This downloads only the small GSM8K text dataset and writes the plan. Run the same
command without `--prepare-only` to generate. `--dry-run` is an alias.
Use local JSONL to plan without network or third-party Python packages:

```bash
python generate.py --input-jsonl examples/tiny.jsonl --num-examples 2 \
  --output outputs/demo --prepare-only
```

Local rows need `question`, `answer`, and optionally a unique `id` containing only
letters/digits/underscores/hyphens. GSM8K-format answers ending in `#### number`
are supported. This is also how to supply manually selected or rewritten questions;
keep wording variants separate from acoustic-only conditions in the analysis.

### Resume

Rerun the **same command**. Completed clips with manifest entries and nonempty WAVs
are skipped; missing clips and failures are retried. No GPU model is loaded if
everything is already complete. Config/plan mismatches are rejected: when changing
count, prompts, sampling settings, voices, or revisions, choose a **new `--output`**.
An interrupted WAV write uses a temporary file so it cannot become a completed clip.
Do not run multiple processes against the same output directory.

## Output and listening

### Replace a delivery after improving its prompt

For a ten-question default run in `outputs/pilot10`, generate angry replacements
and then install them into that dataset:

```bash
python generate.py --num-examples 10 --styles angry --output outputs/angry-v2
python replace_style.py --target outputs/pilot10 --source outputs/angry-v2 --style angry
```

The replacement command requires matching questions, voices, repeats, seeds, and
generation settings. It backs up the old WAVs and metadata under
`outputs/pilot10/backups/`, replaces only angry clips, updates the manifest and
plans, and refreshes the listening page. If `results.jsonl` already exists, it is
archived in the backup and removed from the active dataset because the recordings
have changed. Rerun evaluation afterward. Run replacement only after generation
has finished and while no evaluation is running. This utility needs only Python's
standard library. The other styles, including stressed, retain their recordings.

The resulting dataset combines style prompts from different runs. Use a new
output directory for any future generation; running the original full-generation
command against the merged directory will reject its mixed plan.

```text
outputs/pilot10/
  audio/                  # Native-rate float WAV files, no loudness normalization
  questions.jsonl         # Selected questions, original solutions, numeric answers
  plan.jsonl              # Every planned clip and exact TTS instruction
  run_config.json         # Frozen experiment plan, source metadata, generation options
  environment.json        # Installed versions, GPU, resolved model commit when available
  manifest.jsonl          # Completed clips, relative paths, seeds, duration, RMS, SHA256
  errors.jsonl            # Failure history, if any; previous errors remain after a retry
  listen.html             # Local player, grouped by question, with all voice/style variants
```

Download the output directory and open `listen.html`, or serve it from the GPU host:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory outputs/pilot10
```

Forward port 8000 over SSH (`ssh -L 8000:127.0.0.1:8000 ...`) and open
`http://localhost:8000/listen.html`. No model or web app server is needed to listen.

Each manifest row includes `question_id`, `clip_id`, `text`, `answer`, `source_answer`,
`speaker`, `style`, `repeat`, `seed`, `instruction`, `audio_path`, `sample_rate`,
`duration_seconds`, `peak_amplitude`, `rms`, `generation_seconds`, and `audio_sha256`.
Paths are relative to the output directory, so recordings remain portable to DIFFA.
`answer` is the text after GSM8K's final `####`; no solver prompt is baked into audio.

## Check before interpreting accuracy

- Style names are **requested conditions**, not verified emotion labels. Listen to
  clips, ideally blind to the labels, to check the contrast and speaker consistency.
- Check words, numbers, and full-question completion with listening and/or ASR.
  This project does not automatically transcribe or certify the recordings.
- `possible_token_limit` flags clips near the speech-token cap as a heuristic.
  Inspect these for truncation. Increase `--max-new-tokens` in a new output directory
  if needed; normal completion also does not guarantee correct pronunciation.
- Speech wording is identical across styles in the plan; TTS may still omit,
  repeat, or mispronounce words. Reject entire paired sets or regenerate using a
  predeclared rule independent of the target model's accuracy.
- Save and inspect duration/loudness differences instead of silently normalizing
  them away; they may be part of the delivery effect. FLOAT WAVs preserve amplitude.
- Initial generation seeds match across styles within a question/voice/repeat.
  Different repeats and voices get different seeds. This does not guarantee
  acoustically matched randomness or bit-identical results across GPU/software versions.
- Pin `--model-revision` and `--dataset-revision` to Hugging Face commit hashes for
  a longer experiment. Defaults use `main`; config, source fingerprint, and installed
  versions are recorded, but this is not a fully locked transitive environment.
- Ten questions is an exploratory pilot. Do not select a winning style and report
  its improvement on those same questions as evidence of generalization.

The default prompts control delivery only. Keep the evaluation instruction to
DIFFA constant and never pass its TTS style instruction as part of the input.
For the acoustic-only experiment, the problem itself should arrive via audio;
giving DIFFA the full transcript as well is a separate control condition.

## Development checks

No GPU, model download, or third-party packages required:

```bash
python -m unittest discover -s tests -v
python -m py_compile generate.py
bash -n setup_vast.sh
bash -n setup_diffa.sh
```

These validate planning, paired seeds, sampling, resume/failure behavior, and
metadata handling with a stub backend. Actual CUDA synthesis must be smoke-tested
on the GPU host using the one-question command above.

## Sources and licensing

Code in this repository is MIT-licensed. GSM8K is separately released under the
[MIT license](https://github.com/openai/grade-school-math). Qwen3-TTS code and model
are released under Apache 2.0; see its [repository/model card](https://github.com/QwenLM/Qwen3-TTS).
Generated datasets and model downloads are excluded from git.
