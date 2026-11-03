# GSM8K vocal delivery pilot

Generate the **same math questions** in seven requested vocal deliveries:
**neutral, confident, hesitant, happy, angry, sad, stressed**.

Uses original [GSM8K](https://huggingface.co/datasets/openai/gsm8k) question text and
[Qwen3-TTS-12Hz-1.7B-CustomVoice](https://github.com/QwenLM/Qwen3-TTS) to synthesize
new recordings. The TTS reads supplied questions; it does **not** invent problems or
solve them. Neutral audio is generated with the same model and speaker as every
other style. The answer and solution are saved for scoring but never passed to TTS.

This repo generates stimuli, evaluates them with DIFFA-2, and extracts candidate
delivery directions. It does not claim that requested emotions were realized or
that the extracted directions improve reasoning.

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
Use a new results filename when changing the prompt or generation settings.
Reruns skip clip IDs already saved in that results file and append new answers.
Use a separate filename from emotion classification results, for example
`--results-file results-reasoning-v2.jsonl`.

At the end, evaluation prints correct/total, accuracy percentage, and parse
failures for each style. You can also summarize completed results during or after
a run, without loading a model:

```bash
python summarize.py outputs/pilot10/results-reasoning.jsonl
```

The summary counts the latest saved result per clip ID, so appended reruns are
not counted twice. Parse failures count as incorrect; skipped questions are not
in the denominator. While evaluation is incomplete, styles may have different
counts; use the finished run for a matched comparison. Keep different model or
prompt configurations in separate results files.

## Generate a pilot

### Native emotion-control smoke: IndexTTS-2.5

This separate smoke test uses the model's documented eight-dimensional emotion
vector, without inline tags or added remarks. It generates six recordings of
the same question: a baseline with no explicit emotion vector, then happy,
angry, sad, afraid, and disgusted at requested strength 0.8. All six use one neutral Ryan recording
as their speaker reference and seed 42. The vector strength is a conditioning
weight, not a calibrated emotion score.

```bash
bash setup_index.sh
.venv-index/bin/python index_smoke.py
```

Setup provisions Python 3.11 and IndexTTS in `.venv-index`, separate from the
Qwen, Fish, and DIFFA environments. It downloads `IndexTeam/IndexTTS-2.5` and
skips the optional text-to-emotion Qwen model. Auxiliary encoder/vocoder weights
download on first inference. Optional acceleration/CUDA-kernel builds are disabled
for this smoke test.

Listen at `outputs/index-smoke/listen.html`; WAVs and the plan/manifest are in
that folder. No cropping, filtering, or accuracy evaluation runs. Confident,
hesitant, and stressed are deliberately omitted: they are not native emotion
dimensions of this model. The baseline inherits the reference's delivery;
its perceived neutrality still needs listening.

`--input-jsonl` selects the question file; `--num-examples` defaults to one.
`--styles` selects the native controls to generate. `--reference-path` changes
the reference folder. `--intensity` changes the nonzero vector component; use a
fresh `--output` folder for another strength. `--prepare-only` saves the plan
without importing model/GPU dependencies.

To generate angry and disgusted versions of the same saved 300-question selection
in a separate folder (600 clips):

```bash
.venv-index/bin/python index_smoke.py \
  --input-jsonl outputs/manual200-mixed/source.jsonl --num-examples 300 \
  --styles angry disgusted --output outputs/index300-angry-disgusted
```

Add `--prepare-only` to inspect the plan without loading the model. Question IDs,
original wording, and reference answers are preserved. Use the same command to
resume completed clips. For an Index-specific baseline comparison, explicitly
include `neutral` in `--styles` too; it is not added automatically.

### Extract delivery directions

`extract_vectors.py` is separate from evaluation. It uses the same DIFFA-2
loading and solving prompt, and records decoder block outputs for up to 512
actual forwards. At every layer/pass it averages four position groups:
acoustic, semantic, combined audio, and answer. Layer indices are zero-based;
measurements are before the model's final layer norm.

It selects angry/happy/confident clips classified NO (emotional) and requires
the matching neutral recording to be classified YES (neutral). The binary
classifier does not verify the specific emotion. Matching uses question ID,
speaker and repetition. Both recordings run independently; only forward indices
reached by both contribute. A forward index is not a shared noise level or
necessarily the same answer-completion stage under dynamic decoding.

```bash
.venv-diffa/bin/python prepare_vector_smoke.py
.venv-diffa/bin/python extract_vectors.py --path outputs/vector-smoke/cropped

# Full both-pass extraction:
.venv-diffa/bin/python extract_vectors.py --path outputs/manual200-mixed/cropped
```

The output is `style_vectors.pt` in the input folder's parent. It saves float32
`sums` and `counts` per style/layer/pass/group, plus `pair_mean_sums` and
`pair_counts` that average passes within each question before averaging questions.
Directions have the sign **style minus neutral**. It stores aggregates, not
individual clip activations, and saves only after completion. There is no resume
or automatic train/test split in extraction. Inputs must be cropped clips with
eligible <=30-second baselines, as selected by the classification pipeline.

### Test steering on new neutral questions

`steer.py` reuses the extractor's loader, leaving extraction/evaluation unchanged.
It adds a raw mean-difference vector after one selected decoder block, at the
chosen positions. `--groups` and `--layers` are separate experimental conditions,
not simultaneous interventions. The baseline is run once per clip. A strength
of 1 adds one raw mean shift; 0.5 adds half, and negative strengths subtract it.
Vectors are not normalized automatically.

Before classifying newly generated questions, preserve the classification file
used during extraction (do this once, and keep this snapshot unchanged):

```bash
cp outputs/manual200-mixed/cropped/results-neutral-binary.jsonl \
  outputs/manual200-mixed/extraction-labels.jsonl
```

The script excludes every question in this file, samples unseen Ryan neutrals
with cropped duration <=30 seconds, and never selects questions by correctness.
Crop the new recordings first. Start with two questions to check the real GPU
hook; layer 15 is an initial choice, not a validated best layer:

```bash
.venv-diffa/bin/python steer.py \
  --path outputs/manual200-mixed/cropped \
  --vectors outputs/manual200-mixed/style_vectors.pt \
  --exclude-results outputs/manual200-mixed/extraction-labels.jsonl \
  --num-examples 2 --style angry --layers 15 \
  --groups acoustic answer --strengths 0 0.5 -0.5 \
  --output outputs/steer-angry-smoke
```

This runs five conditions per question: baseline and +/-0.5 for each of two
position groups. For 50 development questions, use `--num-examples 50` and
`--output outputs/steer-angry-dev`. It saves a frozen question plan, full responses,
strict existing answer scores, actual forward counts, and a paired summary.
Rerunning the same command skips completed clip/condition combinations; new
crops do not change the saved sample. `--prepare-only` writes the plan without
loading GPU dependencies.

The default `--mode fixed` uses `pair_mean_sums / pair_counts`: one direction per
layer/group, applied at every forward. `--mode per-pass` uses the corresponding
`sums / counts` direction at each forward. Unsupported passes are left unchanged;
`--min-pairs` sets how many extraction pairs must support a per-pass direction.
The saved vector file supplies the original solving prompt and answer length.
All conditions for a clip use the same random seed. Parse failures remain
incorrect; full answers are retained for review. Reported p-values are exact,
two-sided McNemar tests and are unadjusted during exploratory sweeps.

After choosing layer/group/strength on development questions, evaluate the fixed
choice on different questions. Pass both `extraction-labels.jsonl` and the dev
`results.jsonl` to `--exclude-results` to exclude extraction and tuning questions.
A 50-question sweep is exploratory, not proof of an accuracy improvement. This
script currently tests audio inputs; text-only transfer is a separate experiment.

For a larger sweep, `prepare_steering_exclusions.py` reconstructs the exact
questions that contributed to both-pass extraction, using the unchanged
`extraction-labels.jsonl` snapshot and the cropped manifest. Classification-only
questions did not train the vectors and need not all be excluded. For our
current vectors, expect 55 angry, 42 happy and 35 confident pairs spanning 71
distinct questions. Keep any questions used for layer selection excluded too:

```bash
.venv-diffa/bin/python prepare_steering_exclusions.py
.venv-diffa/bin/python steer.py \
  --path outputs/manual200-mixed/cropped \
  --vectors outputs/manual200-mixed/style_vectors.pt \
  --exclude-results outputs/manual200-mixed/vector-extraction-questions.jsonl \
    outputs/steer-angry-all32-dev/results.jsonl \
  --num-examples 200 --style angry --layers {15..25} \
  --groups acoustic answer --strengths 0 0.5 \
  --output outputs/steer-angry-L15-25-dev200
```

This is 200 x (1 baseline + 11 layers x 2 groups) = 4,600 generations.
Use `--prepare-only` first to verify availability and freeze the sample without
loading the model. These questions are separate from extraction and the prior
10-question layer sweep; some may have appeared in earlier delivery evaluations.
To compare another layer range on exactly the same recordings, pass
`--questions-from outputs/steer-angry-L15-25-dev200-a1/run_config.json` with a
new output folder and the same exclusions/count. This preserves question order
even if more crops arrive. It expands the development sweep; it is not a fresh
confirmation set. Existing runs without this option retain their resume settings.

### Sequential strength, combination and semantic follow-up

`run_steering_followup.py` is a separate runner; it imports the existing hook
from `steer.py` without modifying that script. Its default plan reuses the same
200 questions from `outputs/steer-angry-L15-25-dev200-a1/run_config.json`.
Every intervention uses the fixed angry direction for its own layer and group.

First, it runs one baseline and these six conditions per question:

- L4 acoustic, +0.5.
- L2 answer, +0.5.
- L18 answer, +0.5.
- L20 answer, +0.5.
- L4 acoustic AND L18 answer simultaneously, +1 at each.
- L18 answer AND L20 answer simultaneously, +0.5 at each.

After that stage finishes for all questions, it tests semantic positions at each
layer 0–25 separately, strength +1. The baseline is reused across stages. This
is 1,400 first-stage plus 5,200 semantic generations: 6,600 total. It keeps the
model loaded once, prints stage/question/condition progress, saves full answers
and per-layer hook counts, and resumes completed clip/condition combinations.
Generation times are recorded. Each stage also saves an interim summary.

```bash
# A one-question GPU smoke, including both combinations and two semantic layers:
.venv-diffa/bin/python run_steering_followup.py \
  --num-examples 1 --semantic-layers 4 18 \
  --output outputs/steer-angry-followup-smoke

# Full sequential run, surviving terminal disconnects:
nohup .venv-diffa/bin/python -u run_steering_followup.py \
  > outputs/manual200-mixed/steer-angry-followup200.log 2>&1 < /dev/null &
```

Output defaults to `outputs/steer-angry-followup200/`. Add `--prepare-only` to
inspect the frozen plan without loading the GPU model. `--semantic-layers`
can explicitly select another layer range. Do not run two GPU model jobs if
they do not fit together. This extends development on the same question set;
confirm any selected setting on separate questions afterward.

For the subsequent stronger-direction test, use the separate runner's
`--plan higher-strength`. This reuses the same saved questions and fixed angry
directions, running baseline plus L4 acoustic +2, L18 answer +2, L20 answer +2,
L4 acoustic/L18 answer simultaneously at +2 each, and L18/L20 answer
simultaneously at +1 each. It does not run semantic steering. For 200 questions
this is 1,200 generations. Use a fresh output folder:

```bash
nohup .venv-diffa/bin/python -u run_steering_followup.py \
  --plan higher-strength \
  --output outputs/steer-angry-higher-strength200 \
  > outputs/manual200-mixed/steer-angry-higher-strength200.log 2>&1 < /dev/null &
```

### Alternative TTS smoke: Fish Audio S2-Pro

This separate smoke test speaks the first question from `examples/tiny.jsonl`
in the seven requested styles, using Fish's inline emotion tags. The question
wording is identical, without added remarks. All styles share the same neutral
Ryan recording as a voice-cloning reference and seed 42. Voice cloning may still
vary across styles; listen to check both delivery and speaker consistency.

On the Vast instance, set up Fish in its own `.venv-fish` environment:

```bash
bash setup_fish.sh
.venv-fish/bin/python fish_smoke.py
```

The setup uses Python 3.12, installs audio system dependencies, and downloads
`fishaudio/s2-pro`. It does not modify `.venv` or `.venv-diffa`. Fish's
[documented GPU requirement](https://speech.fish.audio/install/) is at least
24 GB; check free VRAM before running beside the ongoing TTS job.

Outputs: `outputs/fish-smoke/audio/`, `manifest.jsonl`, `plan.jsonl`, and
`listen.html`. No cropping, classification, or accuracy evaluation runs.
One style tag is repeated before each sentence; neutral uses no style tags.
Fish's [tag guide](https://fish.audio/blog/fish-audio-s2-fine-grained-ai-voice-control-at-the-word-level/)
describes tags as applying until the next tag or sentence end. Tags are editable
in `configs/fish_styles.json`; use a new `--output` folder when changing them:

```bash
.venv-fish/bin/python fish_smoke.py --output outputs/fish-smoke-single-tags
```

Change the question with `--input-jsonl` (first row only)
or the voice reference folder with `--reference-path`. The default reference is
the first completed neutral clip under `outputs/manual200-mixed/recordings`.

`--prepare-only` saves the plan without loading Fish or GPU dependencies.

### Classify cropped styles and keep passing clips

Run this separately from generation/cropping, using the DIFFA environment:

```bash
.venv-diffa/bin/python filter.py \
  --path outputs/manual200-mixed/cropped
```

It classifies all seven requested styles, saves every prediction in
`cropped/results.jsonl`, and copies accepted audio plus its manifest to
`outputs/manual200-mixed/filtered/`. All neutral baselines are kept, even if the
classifier predicts another label; emotional clips are kept only when the label
matches the requested style. Source cropped WAVs remain intact.

Available clips are processed without waiting for all seven styles. Compare each
retained emotional style with its matching neutral baseline by question ID.
Your existing duration rule excludes a question if any available recording exceeds
30 seconds. Reruns skip classifications already recorded in the results file.
Use the same results filename to resume. If you used `emotion-results.jsonl`
with the previous version, pass `--results-file emotion-results.jsonl` to reuse it.
Changing classifier prompts/models requires a separate results file and output
dataset; existing decisions are reused by clip ID.

The original default generation cap is 512 tokens. This command loads
DIFFA on the GPU, so inspect VRAM availability before running alongside TTS.
Classifier agreement is a selection rule, not independent confirmation of emotion.

### Generate 50 questions per style, then immediately evaluate

On an instance where both environments have already been set up:

```bash
bash run_experiment.sh 50 outputs/pilot50-v2
```

This generates **350 recordings** (50 questions × seven styles, Ryan, one render),
then runs DIFFA-2 and prints per-style accuracy. The TTS process exits before
DIFFA loads, releasing its GPU memory. Sampling uses seed 123. If
`outputs/pilot10/questions.jsonl` exists, the original ten question IDs are
excluded automatically. Keep this file present when resuming generation.
You can supply another seed as the third argument. Direct generation also accepts
`--exclude-questions path/to/previous/questions.jsonl`.

Run inside `tmux` for a long SSH session. If generation fails, the pipeline stops
before evaluation; rerunning resumes generation. If evaluation is interrupted,
rerunning currently reevaluates the clips and appends results; the summary counts
the latest entry per clip. Generation logs are saved beside the output directory
as `outputs/pilot50-v2.generation.log`; evaluation logs are inside it.

Evaluation uses a concise-reasoning prompt, a **512-token** response budget, and
the `scalar-final-v2` scorer. It tolerates Markdown around the explicit final
answer, numeric units, thousands separators, fractions, and simple boxed answers.
Multiple component counts are flagged for review instead of adding them or
choosing a number based on the reference. Numbers in the reasoning are never
used as a fallback. Responses also record re-encoded token counts and a
`near_token_limit` heuristic; this is not a definitive EOS/truncation signal.
All styles use the same extraction rule and generation settings.

Results go to `outputs/pilot50-v2/results-reasoning-v2.jsonl`. **Fifty questions
are generated; fewer may qualify for evaluation** because a question is excluded
if any of its recordings exceeds 30 seconds. The summary displays the actual
denominator. Original pilot recordings and results are left intact.

### Add 150 new questions, excluding the previous 50

```bash
bash run_experiment.sh 150 outputs/extra150-v2 456 \
  outputs/pilot50-v2/questions.jsonl
```

The fourth and subsequent arguments are previous `questions.jsonl` files whose
IDs must be excluded. The original `outputs/pilot10/questions.jsonl` is also
excluded automatically if present. Missing explicitly supplied files cause an
error before generation; no overlap is silently permitted. This adds **1,050
recordings**, bringing the two larger batches to 200 generated questions per
style before duration filtering. Keep these prior manifests available when
resuming. The evaluation prompt, scorer, voice and 512-token budget are unchanged.

Combine the two larger batches' results after evaluation:

```bash
python summarize.py \
  outputs/pilot50-v2/results-reasoning-v2.jsonl \
  outputs/extra150-v2/results-reasoning-v2.jsonl
```

The initial pilot10 used a different protocol and should not be pooled into this
summary. Direct generation accepts multiple exclusions in one option:
`--exclude-questions previous1/questions.jsonl previous2/questions.jsonl`.

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

### Prepare handwritten style remarks separately

[configs/remarks.json](configs/remarks.json) contains two templates per emotional
style, with `{question}` marking where the complete original question is inserted.
Neutral uses only `{question}`. Each recording uses one template; remarks can
come before or after the question. The answer is preserved.
`prepare_text.py` uses only Python's standard library: no model,
GPU, TTS, or accuracy evaluation runs during text preparation.

```bash
python prepare_text.py --style angry --output outputs/manual-angry
.venv/bin/python generate.py \
  --input-jsonl outputs/manual-angry/tts-input.jsonl \
  --num-examples 1 --styles angry --output outputs/manual-angry/audio-run
```

The default input is the short apples question in `examples/tiny.jsonl`.
Change it with `--input-jsonl your-questions.jsonl`; rows need `question` (or
`text`) and `answer`. Existing `questions.jsonl` files are supported.
Use `--num-examples N` to prepare more questions. Templates alternate across
questions; `--remark-index 1` starts with the second template for a smoke test.
Change the style in both commands to try another delivery.
The single `outputs/manual-angry/tts-input.jsonl` file saves the original question
and prepared wording under `style_texts`. The generator selects the matching
style's text. Listen to the WAV under `audio-run/audio/`.
Text preparation remains separate from `run_experiment.sh` and `evaluate.py`.
Use new output directories when trying another wording. Comparisons now test
wording plus delivery, rather than vocal delivery alone.

For all seven styles, prepare with `--style all` and omit `--styles` when running
`generate.py`. They share one output folder, so evaluation can exclude long
questions consistently across every style.

### Crop off the added remarks

`crop_questions.py` uses the full known spoken text and Torchaudio's
[MMS forced aligner](https://docs.pytorch.org/audio/2.6.0/generated/torchaudio.pipelines.Wav2Vec2FABundle.html)
to locate the original question. It saves separate WAVs and a manifest; source
audio and results stay untouched. It requires recordings from the prepared-text
workflow, with `original_question` in their manifest.

Start with a listening smoke test on the recordings already completed:

```bash
.venv-diffa/bin/python crop_questions.py \
  --path outputs/manual200-mixed/recordings \
  --output outputs/manual200-mixed/cropped \
  --limit 14
```

No new packages are required in the existing DIFFA environment. The aligner
downloads about 1.2 GB of weights on first use. CPU is the default; add
`--device cuda` to use the GPU when available. This is separate from TTS and
does not run evaluation or emotion classification.

Listen through `outputs/manual200-mixed/cropped/listen.html`. Once the cuts
sound good, repeat the command without `--limit` to process all recordings
currently in the source manifest. Existing crops are skipped, and later runs
pick up newly generated clips.

Only alignment audio is resampled/downmixed. Saved crops retain original sample
rate, channels, and samples, without normalization or fades. Neutral WAVs are
copied unchanged. Cuts allow up to 40 ms of padding into the gap between remark
and question (`--padding-ms`), and preserve outer silence where no remark exists.
Numbers use the aligner's wildcard token rather than guessing their spoken form.
The output manifest keeps the uncropped text, source hash, crop boundaries, and
alignment boundary scores. Alignment failures go to `crop_errors.jsonl` and are
not silently replaced with guessed cuts.

Forced alignment assumes the supplied words were spoken; it does not verify
that TTS read the full question correctly. Review the first crops for remaining
remark sounds or clipped question words. The cropped manifest contains the
original question as `text`, ready for later classification/evaluation.

To combine the 130 previously eligible questions with 70 new questions:

```bash
bash run_manual_experiment.sh 200 outputs/manual200-mixed
```

This first reads previous WAV headers and excludes any question with a recording
over 30 seconds. The original two batches retain 130 questions. It then samples
70 new GSM8K test questions, excluding all 200 original IDs, to reach the requested
total of 200 questions / 1,400 clips. New-question sampling uses seed 789.
Neither new questions nor the added remarks are guaranteed to fit 30 seconds;
evaluation checks all new recordings and excludes whole questions as before.
The selected old and new questions are saved together in `source.jsonl`.

It runs text preparation, speech generation, and GPU cropping sequentially,
then stops before emotion filtering or accuracy evaluation. It stops if a step
fails and saves a log for each stage. Rerunning the same command preserves the
saved question selection and skips completed recordings and crops. It uses the question
files and recordings from `pilot50-v2` and `extra150-v2`. To supply other previous
question files, put their paths after the output directory; their folders must
also contain the old manifest and audio. All stages remain separate commands.
To extend the current 200-question run to 300, stop the active runner with Ctrl-C,
then rerun in the same folder with the larger count:

```bash
bash run_manual_experiment.sh 300 outputs/manual200-mixed
```

This appends 100 unseen questions to `source.jsonl`, preserving the previous 200
and their template assignments. The generation plan extends to 2,100 clips;
completed clips are skipped. The `--extend` option used by the runner rejects
changes to existing task text, seeds, or model settings. Cropping runs after all
speech generation completes, then the runner stops for your filtering step.

For a retained-only run, omit `--add-new` when invoking `filter_questions.py`.
For example, to run the retained-only stages separately:

```bash
mkdir -p outputs/manual130
.venv/bin/python filter_questions.py \
  --sources outputs/pilot50-v2/questions.jsonl outputs/extra150-v2/questions.jsonl \
  --output outputs/manual130/source.jsonl
python prepare_text.py --input-jsonl outputs/manual130/source.jsonl \
  --style all --num-examples 130 --output outputs/manual130/text
.venv/bin/python generate.py \
  --input-jsonl outputs/manual130/text/tts-input.jsonl \
  --num-examples 130 --output outputs/manual130/recordings
.venv-diffa/bin/python crop_questions.py \
  --path outputs/manual130/recordings --output outputs/manual130/cropped --device cuda
```

This alternates the two templates within each emotional style. Neutral has no added words.
After inspecting/filtering the cropped recordings, run evaluation separately:

```bash
.venv-diffa/bin/python evaluate.py --path outputs/manual130/cropped \
  --max-new-tokens 512 --results-file results-reasoning-v2.jsonl
```

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
- In the delivery-only condition, speech wording is identical across styles; TTS may still omit,
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

## External emotion agreement on neutral recordings

Two separate entry scripts compare dedicated emotion models against saved DIFFA
binary labels. They select only `style == "neutral"` clips from the cropped
manifest with a DIFFA `yes` (neutral) or `no` (emotional) classification. Selection
does not depend on math answers, an angry partner, or the external prediction.

Use a separate environment so DIFFA/TTS dependencies are not changed:

```bash
bash setup_emotion.sh
.venv-emotion/bin/python emotion2vec_agreement.py --limit 2
.venv-emotion/bin/python sensevoice_agreement.py --limit 2
```

After the smoke runs, omit `--limit` to process all selected neutrals:

```bash
.venv-emotion/bin/python -u emotion2vec_agreement.py
.venv-emotion/bin/python -u sensevoice_agreement.py
```

Defaults read `outputs/manual200-mixed/cropped/manifest.jsonl` and
`results-neutral-binary.jsonl` in the same folder. Override `--path` for another
cropped dataset. Both scripts use CUDA by default; `--device cpu` is supported.
Models download on first use. Each entry prints progress, appends predictions per
clip, and skips saved clip IDs on rerun, including after a smoke run.

Predictions are saved separately as `results-emotion2vec-neutral.jsonl` and
`results-sensevoice-neutral.jsonl`, with adjacent `.summary.json` files. Add
`--summary-only` to recalculate agreement without loading models. No clips are
copied, removed, or filtered by these scripts.

The shared `emotion_agreement.py` reports a 2x2 table, agreement within each DIFFA
group, overall agreement, and Cohen's kappa. Neutral maps to neutral; explicit
emotion classes map to emotional. Other/unknown or missing emotion tags remain
unresolved and are counted separately, excluded from agreement denominators.
Emotion2vec uses the highest-scoring original class and saves all class scores.
SenseVoice uses the raw emotion token before transcription postprocessing, without
VAD segmentation, and saves the raw tagged text. Agreement is not an accuracy
estimate: DIFFA is a comparator, not a human reference.

Official inference examples: [emotion2vec+ Large](https://huggingface.co/emotion2vec/emotion2vec_plus_large)
and [SenseVoice Small](https://huggingface.co/FunAudioLLM/SenseVoiceSmall).

For a third comparison, `speechbrain_agreement.py` uses the official
[SpeechBrain wav2vec2 IEMOCAP classifier](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP).
Reuse the working emotion environment and its installed PyTorch:

```bash
.venv-emotion/bin/python -m pip install -r requirements-speechbrain.txt
.venv-emotion/bin/python -u speechbrain_agreement.py --limit 2
.venv-emotion/bin/python -u speechbrain_agreement.py
```

It uses the same neutral-only selection and reporting, saves original class
scores and labels to `results-speechbrain-neutral.jsonl`, and resumes saved clips.
The four checkpoint labels (neutral, happy, angry, sad) are decoded using its own
label encoder. This classifier has no explicit unknown class. Downloaded model
files are kept under `external/speechbrain-emotion` and the Hugging Face cache.

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
