# DIFFA happy overnight queue

Run on the DIFFA machine (`C.57998769`), using `.venv-diffa`. Export `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` in the launching shell; the script never prints them.

```bash
git pull --ff-only
nohup bash run_diffa_happy_nightly.sh > outputs/diffa-happy-nightly.log 2>&1 </dev/null &
tail -F outputs/diffa-happy-nightly.log
```

Default queue, across GSM8K, SVAMP, MultiArith, BBH3, BBH5 and IFEval:

1. Prepare missing Jenny happy audio and collect happy-minus-neutral vectors.
2. Compare unsteered happy delivery against existing neutral responses.
3. Steer neutral with its own happy vector at L31 answer×+1, then stop.

**No signed-strength sweep or transfers are scheduled by default.** These require explicit `--stages` options in a separate request. The default strength list is only+1.

One DIFFA model is loaded after waiting for the shared GPU lock and active GPU work. Tasks run sequentially within that process; no model training/checkpoint changes. Finished pairs and answers are skipped on resume. An output queue lock prevents duplicate writers. Unrelated legacy audio plan entries without question IDs/text/audio paths are ignored during inventory; requested task recordings still must pass the frozen question/audio checks.

## Questions, prompts, and neutral reuse

Original sources are `winning_diffa_catalog.json`. Each new direction uses the **exact extraction question IDs recorded by its angry vector**, and evaluates on the same disjoint target IDs. The historical GSM vector was classifier-selected; its existing selection is inherited, not reapplied to happy. Other source cohorts are likewise preserved. The original collection prompt is preserved separately from the original target prompt; in particular GSM's historical collection used `Final answer:` while its held-out target prompt uses `####`. This differs from the older, separate `run_diffa_gsm_happy.sh` draft, which used all500 excluded IDs and the aligned prompt for both phases. The two vector protocols are not interchangeable.

IFEval keeps1024 answer positions/tokens; the other tasks keep512. All use seed42 reset per generation, fast_dllm,128 steps, block32, temperature0. Full unchanged audio follows the existing native≤30s/chunked Whisper+global-adapter path above30s, without cropping. Historical IFEval eligible question IDs remain frozen; if a newly generated happy clip exceeds30s it is explicitly logged and handled through the full chunked path, not silently removed from the cohort.

Held-out **neutral responses are never regenerated**. Exact compatible neutral answers must exist in the original DIFFA runs; otherwise preflight stops before model loading. Existing happy audio is reused when Jenny voice, text and reference match; only missing happy clips are generated with Azure Jenny, intensity1.5. Compatible previous happy answers are reused when prompt, audio hash, seed and decoding settings match. Older happy responses under a different prompt are not relabeled as new results. Ryan and Guy audio and classifier predictions are not used in this queue.

Collecting the new vector still requires passive **neutral collection forwards**, paired with happy forwards. Those are distinct from rerunning neutral answers on the held-out test split. Both raw collection responses are saved separately.

## Vectors and outputs

For every collection pair, record all32 decoder layers and four groups: acoustic, semantic, combined audio, and answer. Token means are averaged over the first min(neutral forwards,happy forwards) in both runs, then over questions with equal weight. Vectors have the sign **happy−neutral** and retain raw magnitude. Positive steering adds that vector to neutral answer positions at L31 on every forward; unsteered happy delivery is a separate condition.

Under `outputs/diffa-happy-winners`:

- `<task>/run_config.json`: full collection/test IDs, audio hashes, prompts, budgets, original source receipts.
- `<task>/collection-stream.pt`: atomic, resumable running sums plus raw responses and per-pair drift scalars. No new full per-question activation cache is written. Existing `pairs/*.pt` files are reused and left untouched.
- `<task>/collection-report/`: live/complete saved collection response comparison, scoring references, lengths and paired metrics. Explicitly labelled collection; kept separate from held-out delivery/steering.
- `<task>/collection-storage.json`: explicit storage policy. Full per-question activation tensors cannot be reconstructed from the aggregate checkpoint; final all-layer vectors, drift statistics and raw answers are retained.
- `<task>/style-vectors.pt`, `layer-drift.csv`, `collection-results.jsonl`: all-layer directions, drift and collection raw text.
- `<task>/delivery/{results,rescored}.jsonl`, `summary.json`: neutral versus happy, held-out only.
- `<task>/strength-1/`: neutral versus own happy-vector+1 steering.
- Additional strengths/transfers are opt-in only; the default run creates neither.
- Root queue manifest, code hashes and runtime model configuration.

Summaries use the same existing upstream saved-answer scorers as the angry experiments: GSM8K filters on math references, BBH task scoring, official IFEval checkers. Numeric equivalence, SVAMP author tolerance, and math-format extractability remain separate diagnostics. Mean/median response tokens exclude special tokens and are retokenized when older results lack counts. Paired McNemar tests and95% question-paired bootstrap intervals are exploratory and unadjusted. No emotion-classifier filtering or new reasoning-versus-no-reasoning contrast is included.

## Commands

```bash
# Saved collection comparisons (including partial checkpoints) and held-out results, separately labelled; CPU only.
bash run_diffa_happy_nightly.sh --mode summary

# One dataset's results.
bash run_diffa_happy_nightly.sh --mode summary --datasets gsm8k

# Explicitly state the default scope if desired.
bash run_diffa_happy_nightly.sh --stages primary
```

The default command and `--stages primary` are the same run; do not launch both. The existing angry output folders remain unchanged.

```bash
# Evidence after/during the queue: results, prompts, vectors, summaries and samples.
.venv-diffa/bin/python pack_experiment_backup.py --only emotion-steering \
  --destination /workspace --samples 3 --sample-mb 100
```

The default collector keeps float64 running sums of the original float32 pair differences, preserving exactly the full-cache mean vector and drift calculation without lossy quantization. It atomically checkpoints after every question pair, so interruption repeats at most the uncommitted pair. The disk preflight reserves approximately0.6GiB for checkpoints/results rather than11.6GiB for all pair tensors. Existing generated audio/configurations/results are reused on restart. The standard backup includes aggregate checkpoints, root vectors, raw text/configuration and selected audio samples. CPU tests exercise budgets/position spans, vector sign, exact neutral reuse, missing-baseline failure, frozen splits and transfer leakage. Actual Azure/GPU inference runs on the remote machine, not in local tests.

The summary reads saved collection responses from the running checkpoint or legacy pair caches, so you do not need to wait for held-out happy generation to see the collection comparison. It performs no inference and does not restart collection. Collection accuracy is descriptive because these questions fit the vector. For GSM, the historical collection prompt may request `Final answer:` while held-out evaluation requests `####`; those prompt cohorts are not pooled, and upstream strict-format extraction on the collection is explicitly flagged as a diagnostic. The report saves the recovered BBH/IFEval reference metadata in `collection-report/scoring-bank.json`.
