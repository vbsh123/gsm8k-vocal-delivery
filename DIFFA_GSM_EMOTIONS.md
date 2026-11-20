# DIFFA GSM8K happy / sad directions

For the newer six-task happy queue that matches the historical angry collection IDs and prompts, use [DIFFA_HAPPY_NIGHTLY.md](DIFFA_HAPPY_NIGHTLY.md). The GSM-only protocol below remains a separate aligned-prompt experiment.

Start with happy versus neutral. No training or model checkpoint changes.

```bash
nohup bash run_diffa_gsm_happy.sh > outputs/diffa-gsm8k-happy.log 2>&1 </dev/null &
tail -F outputs/diffa-gsm8k-happy.log
bash run_diffa_gsm_happy.sh --mode summary
```

The launcher prepares/reuses Azure audio on CPU, then waits for the shared
`outputs/jenny-task-vectors.lock` and active GPU work to finish before loading DIFFA.
It collects first, writes vectors and drift, then evaluates neutral/happy/steered
neutral at L31 answer ×1. Raw responses are saved as each generation finishes.
This script was checked with CPU fixtures; full GPU inference is not run locally.

## Frozen questions and prompts

- Reads `outputs/reasoning-transfer/gsm8k/run_config.json` and the referenced
  original vector. Uses its 500 excluded collection IDs and the original718
  disjoint held-out IDs. It does not pick questions based on correctness.
- Needs `outputs/reasoning-transfer/gsm8k/gsm8k-test.jsonl`, the official question
  source already downloaded by the aligned GSM8K run.
- Prefers original Jenny audio in `azure-jenny500-neutral`, `azure-jenny500-extra`,
  `azure-jenny-next500`, `azure-jenny-remaining-test`. Checks official wording/gold
  and preserves each held-out neutral recording's original SHA256.
- Missing neutral/happy audio alone is generated in the new run's `audio/` folder,
  Jenny intensity1.5. Existing source folders are not rewritten. Missing audio
  generation requires exported `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION`.
- The same aligned `#### <number>` prompt is used for collection and evaluation.
  Older happy responses under `Final answer:` are not reused as aligned results.
- All500 collection pairs contribute, without duration or classifier filtering.
  This differs from the historical angry vector (425 classifier-selected pairs,
  `Final answer:` collection prompt). Preserve this distinction in comparisons.

BF16 model, fast_dllm,128 steps, block32, temperature0,512 answer positions,
seed42 reset before each generation. Existing `gen_chunked` preserves full audio:
native up to30s, chunked Whisper and global connectors above30s; no cropping.

## Collection and analysis

Each full decoding run observes all32 decoder-block outputs. At each forward,
mean acoustic/semantic/combined-audio/answer activations are captured. For each
pair use the first min(neutral forwards, emotion forwards) in both conditions;
mean their difference over forwards, then average the500 pair differences equally.
The saved sign is **happy minus neutral** (or sad minus neutral), with raw magnitude.
Positive steering adds that vector to neutral audio's answer activations on every
forward; the happy delivery itself is unsteered.

All layer/group means and pair raw responses are retained in `pairs/`; allow a
few GiB for these caches. `layer-drift.csv` reports mean/median relative change,
state cosine and shift consistency from matched-forward pooled pair means. These
are descriptive activation diagnostics, distinct from SAME100 initial-forward
reasoning contrasts. This run does not collect a new reasoning-minus-answer-only
contrast or perform cross-task transfer yet.

Raw text: `collection-results.jsonl` and
`evaluations/L31-answer-a1/results.jsonl`. Configs include question bank, exact
prompt, audio hashes, original cohort/vector receipts, and generation settings.
`style-vectors.pt` stores all32 layers/four groups under the actual emotion key.
`runtime.json` records runtime/model configuration and generation-code hashes.

Summary uses installed GSM8K harness strict/flexible extraction, additional numeric
comparison and format diagnostics; exact paired McNemar and200,000-draw paired
bootstrap CIs are exploratory/unadjusted. It includes mean/median retokenized
response tokens excluding special tokens. The same neutral cohort is used, and
comparisons report only completed matched question pairs.

## Other strengths, resume, and backup

A resumed default command skips completed pairs/answers and retains its vector.
Different strength lists get separate immutable evaluation folders, reusing the
same collection/vector and eligible baseline answers. For example, **later**:

```bash
bash run_diffa_gsm_happy.sh --strengths -2 -1.5 -1 -0.5 0.5 1 1.5 2
bash run_diffa_gsm_happy.sh --mode summary --strengths -2 -1.5 -1 -0.5 0.5 1 1.5 2
```

For sad later use `diffa_gsm_emotion.py --emotion sad`; this defaults to a separate
`outputs/diffa-gsm8k-sad` folder. Neither sad nor the extended strength sweep is
launched by the default happy command.

```bash
.venv-diffa/bin/python pack_experiment_backup.py --only emotion-steering \
  --destination /workspace --samples 3 --sample-mb 100
```

Tests: `python -m unittest discover -s tests -p test_diffa_gsm_emotion.py`.
