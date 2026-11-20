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
