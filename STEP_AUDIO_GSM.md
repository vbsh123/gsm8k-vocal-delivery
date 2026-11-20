# Step-Audio-2-mini neutral versus angry

Run on the DIFFA machine:

```bash
git pull
nohup bash run_step_gsm400.sh > outputs/step-gsm400.log 2>&1 </dev/null &
tail -F outputs/step-gsm400.log
```

The launcher validates/fixes the experiment configuration, downloads the pinned
checkpoint on CPU, waits for `outputs/winning-transfer/queue.lock`, then takes
`outputs/jenny-task-vectors.lock` and waits for active GPU processes to finish.
It loads one Step model only after those waits. It does not install packages,
change the existing DIFFA environment, start TTS, collect vectors, or steer.

Protocol:

- Source: `outputs/reasoning-transfer/gsm8k/run_config.json`.
- Fixed 400 question IDs: source cohort order shuffled with seed42, first400.
  No selection by correctness, emotion labels, or duration. All800 recordings
  must already exist with the saved SHA256 and paired voice/reference identity.
- Task prompt copied verbatim from that source configuration (the DIFFA reasoning
  and `#### <number>` prompt); printed at preparation and saved with each answer.
- Official Step system prompt `You are a helpful assistant.` and human/assistant
  chat delimiters; identical for neutral and angry.
- Native full audio input using the official 25-second windowing and log-mel
  helpers. No transcription, custom feature splice, or audio crop.
- BF16, greedy, seed42 reset per generation, 512 new token budget. Native model
  forward/cache behavior is unchanged; upstream returns no KV cache, so generation
  may be slower than other AR models. No TTS prefix or speech decoder/vocoder.
- Raw text answers and generated suffix token counts saved. Count includes EOS
  when generated; decoding removes terminal EOS only if present, retaining the
  last token on budget-limited answers. Official text decoding excludes audio codes.
- Existing upstream GSM8K strict/flexible scoring, numeric-equivalence diagnostic,
  format diagnostic and paired unadjusted statistics. Not full official text-harness
  inference. Mean/median token counts use saved suffix lengths.

Pinned model: `stepfun-ai/Step-Audio-2-mini` at
`e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36`.
Native template/windowing follows official `stepaudio2.py` and `utils.py` at
Step-Audio2 revision `76e272b56c3917a8d7188f18bbb5a65dfc8a0845`.
[Model](https://huggingface.co/stepfun-ai/Step-Audio-2-mini)
[Inference source](https://github.com/stepfun-ai/Step-Audio2/tree/76e272b56c3917a8d7188f18bbb5a65dfc8a0845)

The four required weight shards total16,630,448,936bytes (15.49GiB). Optional
`token2wav/` files are excluded from new downloads. The existing DIFFA environment
already pins Torch/Torchaudio2.6.0 and Transformers4.49.0, matching this adapter.
The downloader checks missing weights plus4GiB reserve; no data are deleted.
Do not uninstall DIFFA dependencies to make room.

Resume with the same command. Configuration changes require a separate output.
For example, use `--num-questions 0 --output outputs/step-gsm718` for the whole
source cohort; it is not the main1269 bank. Saved subset results are not silently
merged or promoted to another cohort.

CPU-only summary (can run while generation continues):

```bash
.venv-diffa/bin/python evaluate_step_gsm.py --mode summary
```

Files:

- `step_audio_adapter.py`: native input assembly and generation.
- `evaluate_step_gsm.py`: fixed pairs, source hashes, GPU queue, shared reporting.
- `prepare_step_audio.py`: dependency imports and pinned download, no GPU model.
- `bbh_candidate_models.json`: pinned checkpoint/settings.
- `outputs/step-gsm400/`: configuration, checkpoint inventory, runtime, raw
  `results.jsonl`, scored responses, summary and scorer provenance.

Evidence archive:

```bash
.venv-diffa/bin/python pack_experiment_backup.py \
  --only step --destination /workspace --samples 3 --sample-mb 100
```

Local tests cover native windows over long recordings, no gold in model inputs,
EOS/budget decoding, fixed paired selection, changed-audio rejection, resume
identity and model loading after GPU wait. No local GPU/weight inference tested.

## Same20 calculation diagnostic

`step_calculations20.py` reuses the first20 IDs in the frozen Step400 bank order,
without selecting by correctness. It requires intermediate arithmetic/equations
before the final answer, using the same wording in text and audio. It runs60 new
generations:20 text questions,20 neutral audio,20 angry audio. The original40
audio answers are copied separately for prompt comparisons, never overwritten.
The original pinned model, native audio template, greedy512-token recipe and
shared GPU wait are preserved. All reported response lengths are retokenized text
without special tokens; generated suffix IDs and EOS flags are also saved.

```bash
nohup .venv-diffa/bin/python -u step_calculations20.py \
  > outputs/step-calculations20.log 2>&1 </dev/null &
tail -F outputs/step-calculations20.log
.venv-diffa/bin/python step_calculations20.py --mode summary
```

This is a small exploratory diagnostic, not a replacement benchmark. No solutions
or reference answers are passed to the model. Text versus audio is a modality
comparison; output length alone does not establish an internal reasoning mechanism.

## Full400 with required calculations

```bash
git pull --ff-only
nohup bash run_step_gsm_calculations.sh > outputs/step-gsm400-calculations.log 2>&1 </dev/null &
tail -F outputs/step-gsm400-calculations.log
bash run_step_gsm_calculations.sh --mode summary
```

Uses the exact frozen400 questions and audio from `outputs/step-gsm400`, with the
same prompt as the20-question calculation diagnostic. Neutral/angry audio only;
no text condition and no new TTS. Existing40 matching audio answers from that
diagnostic are reused, leaving760 new generations when the diagnostic is complete.
The original400-question results are preserved; new results go to
`outputs/step-gsm400-calculations`. Checkpoint, decoder,512-token budget and shared
GPU wait are unchanged. Resume with the same command. The prompt was chosen after
inspecting the20-question diagnostic, so this is an exploratory follow-up; those
questions are not independent confirmation.

## Disjoint Step activation collection

The evaluation runner does not capture hidden states. Queue a separate collection
after it; missing activations cannot be recovered from saved text alone:

```bash
nohup bash run_step_gsm_vectors.sh > outputs/step-gsm400-vectors.log 2>&1 </dev/null &
tail -F outputs/step-gsm400-vectors.log
bash run_step_gsm_vectors.sh --mode summary
```

Default400 neutral/angry pairs (800 additional full generations), sampled with
seed42 from the original500 DIFFA collection exclusions. They are disjoint from
all718 original evaluation questions, including the current Step400. No correctness
or emotion-classifier filtering and no TTS. The calculations prompt, checkpoint,
512-token greedy decoder and native audio path match the current Step evaluation.
The collector waits for that evaluation's run lock and the shared GPU lock/idle.
It never attaches to or changes the already running process.

Passive hooks observe all28 **language decoder** layers (not the32-layer audio
encoder). Audio means use initial prefill audio-patch positions. Answer means
observe the last prediction state once per native forward, including the initial
assistant boundary that predicts the first answer token. Each condition's forward
means are averaged independently, then angry minus neutral is computed per
question and questions receive equal weight.

`outputs/step-gsm400-vectors/` retains:

- `initial-vectors.pt`: first-prefill angry-minus-neutral means for answer/audio.
- `style-vectors.pt`: full-generation mean answer prediction states and initial
  audio states, angry minus neutral, for future steering.
- `pairs/`: both conditions' mean/initial states, raw responses, token IDs, EOS
  flags and forward counts (roughly1.3 GiB at400 pairs).
- `results.jsonl`: raw collection answers, updated after each completed pair.
- `layer-drift.csv`: initial-state mean/median relative change, cosine and shift
  consistency across all layers and both groups.
- Frozen source IDs/audio hashes, runtime and code hashes, and descriptive
  collection scoring. Collection accuracy is not held-out validation.

Capture hooks do not replace block outputs or patch native audio/cache handling.
Initial-state geometry and generation-history means are distinct; the latter can
reflect differing output lengths/content and are not proof of a reasoning mechanism.
Resume skips saved pairs and retains frozen vectors. No steering is launched.

`--only step` includes the vectors, raw responses and summaries. For auditing the
per-pair state means as well, explicitly package:

```bash
.venv-diffa/bin/python pack_experiment_backup.py --only step-vectors \
  --destination /workspace --samples 3 --sample-mb 100
```
