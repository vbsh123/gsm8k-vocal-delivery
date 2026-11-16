# Kimi BBH three-object recovery layer diagnostic

This experiment uses **all 30 flexible-score angry-correct / neutral-wrong pairs**
from the completed regenerated Kimi BBH-three benchmark. There were 30 recoveries
and 24 regressions, hence six **net** additional correct answers; six is not the
number of recovery questions. Selection is computed from saved upstream scores,
not hard-coded IDs. Original answers, audio, and summaries are untouched.

Collection uses **125 matched pairs**, selected with seed42 from the 220 remaining
questions after excluding all 30 recovery test questions. Both angry-minus-neutral
and Instruct-minus-base differences use those SAME 125 collection questions.
The 30 recovery questions are used **only for steering**. The sets are disjoint,
but the test set is outcome-selected: this is a recovery diagnostic rather than
an estimate of overall accuracy benefit.

## Run on the machine with Kimi results and their original audio

```bash
git pull --ff-only
nohup bash run_kimi_bbh_layers.sh \
  > outputs/kimi-bbh-layers.log 2>&1 < /dev/null &
tail -f outputs/kimi-bbh-layers.log
```

The wrapper waits for existing IFEval/IFBench, steering, reasoning, and audio-model
jobs, then uses the shared GPU lock. It uses the existing Kimi environment for
inference and the DIFFA environment for CPU harness scoring. It loads one Kimi
checkpoint on the GPU at a time. The base checkpoint requires approximately
17 GB of additional downloadable weights; the script deletes nothing.

Inputs default to `outputs/bbh-three-regenerated-models/kimi/{run_config.json,rescored.jsonl}`.
If necessary refresh the baseline scored file first, on CPU:

```bash
.venv-diffa/bin/python evaluate_bbh_audio_models.py \
  --model kimi --mode summary --output outputs/bbh-three-regenerated-models
```

## What is collected

1. Reproduce neutral and angry answers for the 125 collection questions with the same pinned
   Kimi-Instruct checkpoint, native greedy decoder, prompt, seed42 and 512-token
   budget. Stop and retain an explicit mismatch report if an answer differs.
2. Capture **both token groups separately at all 28 shared multimodal layers**:
   **audio** means over the native `is_continuous_mask` positions during prefill
   (not the text instruction or special delimiters), and **answer** means over
   the last prediction position across native generation forwards, including
   prefill and EOS prediction. Subtract angry minus neutral within each question,
   then average the 125 questions equally. Keep raw magnitudes; no norm rescaling.
3. Also replay the **actual neutral answer token inputs**, without retokenizing
   the response string, with angry audio. This gives a matched-answer-history
   emotion difference. Audio-prefix offsets may differ between deliveries, but
   the answer tokens and number of prediction steps are identical. An unchanged
   neutral replay must reproduce BOTH native audio and answer activation means.
4. Load pinned `moonshotai/Kimi-Audio-7B` base and replay exactly the frozen
   Instruct-encoded neutral inputs and neutral answer histories. Instruct minus
   base is a **decoder-checkpoint difference**, holding audio encoder features
   fixed. Base free generation and independently re-encoded base audio are not
   part of this comparison. The raw mean checkpoint difference is saved and
   compared separately for AUDIO and ANSWER on those same 125 questions by layerwise cosine. It is never used
   for steering. Checkpoint differences include post-training effects broadly;
   similarity alone does not demonstrate an alignment mechanism or a causal role.
5. Reload Instruct and sweep each layer and token group independently at x1:
   **audio-vector to audio positions** during prefill only, and **answer-vector to
   the last prediction position** during prefill and cached answer steps.
   Audio steering does not modify instructions or answer slots; cached steps have
   no input audio positions. For 30 recoveries this is **1,680 steered answers**
   (30 × 28 × 2), plus collection/replay passes. The six speech-output branch
   layers do not feed text logits and are excluded. Post-block audio steering at
   the final shared layer has no downstream text-attention block to propagate
   through, so that row is an expected no-effect control, not an implementation
   failure.

All capture points are decoder block outputs **before** final RMS normalization.
Kimi is autoregressive: these are cached next-token prediction states, not
DIFFA denoising steps or its simultaneously masked answer-token block. The native
Kimi audio pipeline handles the full recordings without the custom DIFFA encoder.

## Outputs and resume

Everything goes under `outputs/kimi-bbh3-collect125-audio-answer-steer30/`:

- `run_config.json`: selection, original responses, audio hashes, prompt, model
  and tokenizer/source revision pins, collection definition and intervention.
- `pairs/*.pt`: resumable per-question means, native neutral input token trace,
  frozen neutral encoded features, and base comparison data. These are small
  activation caches, not model weights.
- `style-vectors.pt`: all 28 audio, native-answer, and matched-history-answer
  emotion mean differences, stored under distinct keys.
- `checkpoint-differences.pt` and `layer-cosines.csv`: separate audio/answer
  checkpoint comparisons; CSV has 56 rows, with explicit group and layer columns.
- `results.jsonl`: raw steered responses and original unsteered references.
- `rescored.jsonl`, `summary.json`, `harness/`: installed upstream BBH scoring,
  strict/flexible correctness and the extra strict-format-extractability diagnostic.

Rerunning resumes complete per-question collection files and missing individual
layer/question answers. Settings changes require a different output directory.
Mid-sweep CPU report (no GPU/model loading):

```bash
.venv-diffa/bin/python kimi_bbh_layer_diagnostic.py --mode summary
```

The report intentionally gives counts/percentages without inferential p-values:
all selected test baseline answers failed flexible scoring and all angry references
passed. Those questions do NOT supply the collected vectors. Layer differences also
reflect varying raw vector magnitudes; a large cosine or sweep gain does not by
itself identify a uniquely important emotion layer.

Modes `prepare`, `collect`, `compare`, and `sweep` can be run separately with the
Kimi Python, in that order. `all` is the default. `--max-generations 4` limits only
the steering sweep for a resumable smoke check. All original experiments remain
separate from this diagnostic.

The obsolete runs are preserved separately under
`outputs/kimi-bbh3-recovery-layer-diagnostic/` and
`outputs/kimi-bbh3-collect125-steer30/`. Their answer-only caches and vectors are
never reused by this audio+answer run. Stop the old diagnostic before pulling
and restarting the wrapper.
