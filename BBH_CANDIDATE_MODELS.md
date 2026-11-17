# Three optional BBH audio models

Prepared choices: **Phi-4-multimodal**, **MiniCPM-o 4.5**, and **Audio Flamingo 3**.
No model is automatically selected, downloaded, installed or run by preparing
this code. Launch ONE command below when ready. Each command sets up only that
model's environment and resumes its own answers.

All three use the same **250 BBH three-object logical-deduction questions**,
neutral and angry Azure Jenny recordings, upstream EleutherAI BBH strict/flexible
filters and exact-match metric. Default is **both prompts**, independently:

- Original: "Let's think step by step."
- Explicit: "Give concise step-by-step reasoning with only the necessary deductions."

The rest of the prompt and final `The answer is (<letter>).` instruction are
identical. There is **no steering or vector collection** in these three runs.
Each model runs 250 × 2 deliveries × 2 prompts = **1,000 generations**, seed42,
greedy decoding, 512 new tokens, single beam, repetition penalty1. Answers are
retained verbatim. The primary table uses all matched questions. The old 125
subset printed by the shared scorer is descriptive, not independent validation.
Prompt variants and model results are never pooled as extra independent samples.

## Choose later

Pull the code, then choose ONE:

```bash
git pull --ff-only

nohup bash run_bbh_candidate.sh phi4 > outputs/bbh-phi4.log 2>&1 < /dev/null &
# OR
nohup bash run_bbh_candidate.sh minicpm > outputs/bbh-minicpm.log 2>&1 < /dev/null &
# OR
nohup bash run_bbh_candidate.sh flamingo > outputs/bbh-flamingo.log 2>&1 < /dev/null &
```

Tail the corresponding log, e.g. `tail -f outputs/bbh-phi4.log`.
To run one prompt only append `--prompt original` or `--prompt explicit`.
For a short inference smoke check append `--prompt original --max-generations 6`;
remove the limit to resume the full experiment. Smoke answers remain saved.

Use `--source /path/to/azure-bbh-three-models-regenerated` to select existing
audio explicitly. Otherwise the runner discovers a complete 250-question folder
or resumes generation in `outputs/azure-bbh-three-models-regenerated`. It requires
`AZURE_SPEECH_KEY` and **`AZURE_SPEECH_REGION`** only when clips are missing.
Generating missing audio makes Azure requests; existing identical clips are reused.
All full recordings are preserved, with no 30-second eligibility filter.

CPU preparation freezes configs before any checkpoint download. The queue waits
for current Kimi runs, including BBH-five, and shares the existing GPU lock. It
does not terminate jobs or delete model caches/audio/results. Environments are
separate: `.venv-bbh-phi4`, `.venv-bbh-minicpm`, `.venv-bbh-flamingo`.

## Sizes and native inference

| Choice | Pinned Transformers | Python | Checkpoint disk, approximately |
|---|---|---|---:|
| phi4 | 4.48.2 | 3.10 | 11.93 GiB |
| minicpm | 4.51.0 | 3.10 | 17.46 GiB |
| flamingo | 5.0.0 | 3.11 | 30.80 GiB |

Allow additional space for the chosen environment and at least four GiB reserve.
The checkpoint downloader checks uncached weight estimates against free space;
it never frees space automatically. No need to install all three environments.
All model revisions are frozen in `bbh_candidate_models.json`. Setup pins
Python/Torch/Transformers, runs `pip check` and records its exact package freeze.

- **Phi-4:** native audio processor with `<|audio_1|>` and its standard user/
  assistant wrapper, speech adapters, SDPA. Full mono audio is resampled to16kHz
  before the processor. The card recommends up to40seconds for satisfactory
  performance; BBH may exceed that, so max duration and counts above40seconds
  are saved, without filtering. [Official model card](https://huggingface.co/microsoft/Phi-4-multimodal-instruct).
- **MiniCPM:** native `chat`, processor and audio encoder. `enable_thinking=False`
  and text-only output are fixed across both prompts. Its own audio chat template
  is retained; vision and audio generation are disabled. Token counts are
  retokenized text lengths because native chat returns decoded text.
  [Official model card](https://huggingface.co/openbmb/MiniCPM-o-4_5).
- **Flamingo:** official Transformers conversion `nvidia/audio-flamingo-3-hf`;
  native processor handles its30-second windows. Inputs exceeding the native
  600-second cap are rejected instead of truncated. No AF-Think adapter. Text is
  decoded only from the generated suffix, retaining all ordinary answer text.
  [Official model card](https://huggingface.co/nvidia/audio-flamingo-3-hf).

These are native audio-conditioned model calls, not ASR followed by a text model.
No transcript, gold answer or intended emotion label is included in model input.
All adapters use full audio, bfloat16 and greedy native decoding. Their different
chat templates and architectures are recorded; the controlled comparison is
neutral versus angry **within each model and prompt**.

## Results

```bash
.venv-diffa/bin/python evaluate_bbh_candidates.py --model phi4 --mode summary
.venv-diffa/bin/python evaluate_bbh_candidates.py --model minicpm --mode summary
.venv-diffa/bin/python evaluate_bbh_candidates.py --model flamingo --mode summary
```

Each model has `original/` and `explicit/` beneath
`outputs/bbh-three-candidates/<model>/`. Preserve raw `results.jsonl`, frozen
`run_config.json`, `runtime.json`, scorer provenance, rescored answers and summaries.
Summary reports matched N, strict/flexible accuracy, additional format
extractability, exact McNemar p-values and paired-bootstrap95% intervals. Tests
and intervals are unadjusted; this is a spoken adaptation using installed
upstream scoring, not the full official text-only prompting protocol.

CPU tests cover source selection, prompt parity, adapter input contracts,
full-length resampling, disk guards and resume compatibility. GPU inference has
not been executed locally; run the short smoke command on the target machine
before treating any adapter as GPU-validated.
