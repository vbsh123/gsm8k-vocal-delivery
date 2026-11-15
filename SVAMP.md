# SVAMP: 500-question neutral/angry delivery evaluation

`run_svamp500.py` selects 500 of the official 1,000 SVAMP challenge problems by
shuffling once with seed 42. It preserves Body + Question and the numeric gold
answer, and sends only Body + Question to Azure. Equations and gold answers are
never spoken or supplied to DIFFA. Source revision and SHA-256 are frozen in code;
selected IDs are saved. The other 500 problems remain outside this initial run.

The same Jenny voice reads neutral and angry versions, intensity 1.5. Only pairs
where both recordings fit 30 seconds enter evaluation; nothing is cropped.

By default, only two conditions run on the same eligible questions:

- Neutral audio, no steering.
- Angry audio, no steering.

This script performs no steering or vector collection. The obsolete GSM8K-vector
transfer branch has been removed. Current SVAMP vector collection and steering
on the other 500 questions live in `run_reasoning_transfer.py`; see
`REASONING_TRANSFER.md`.

DIFFA uses the existing shared loader/generator, the identical GSM8K math prompt,
512 output tokens, 128 steps, 32-token blocks, fast-dLLM, temperature zero, and
seed 42 reset before each condition. Only one model is loaded. CPU/cloud TTS runs
first; inference waits for the shared GPU lock and existing GPU processes.

The primary score is **flexible numeric correctness**: the upstream GSM8K harness
extractor, followed by our numeric-equivalence comparison. The report also includes
upstream-style flexible string matching, strict `####` extraction accuracy, and
the rate of extractable `####` numeric answers.
This is not an official SVAMP harness inference task. Saved responses are scored
without using gold answers to choose the extracted prediction. All comparisons
against neutral include paired McNemar p-values and paired bootstrap confidence
intervals; these are unadjusted. Original response files remain untouched.

On the steering machine, with Azure environment variables exported:

```bash
git pull --ff-only
mkdir -p outputs
nohup .venv-diffa/bin/python -u run_svamp500.py --answer-format hashes \
  > outputs/svamp500-hashes.log 2>&1 < /dev/null &
tail -f outputs/svamp500-hashes.log
```

Run only generation with `--mode generate`, or only evaluation with `--mode
evaluate`. Neutral/angry accuracy is the default. Rerunning resumes
saved audio and answers. Configuration changes require a separate output folder.

Check completed results at any time, without GPU work:

```bash
.venv-diffa/bin/python -u run_svamp500.py --mode summary --answer-format hashes
```

Recordings and a listening page: `outputs/azure-svamp500/`.
Raw answers, frozen config, upstream scorer provenance, rescored answers and
summaries: `outputs/svamp500-diffa2-hashes/`.

The legacy `Final answer:` prompt remains available with `--answer-format
final-answer`, using `outputs/svamp500-diffa2/`. Summary mode can still read old
saved responses and ignores old steering-condition rows. Original raw responses
are not rewritten. The obsolete `--steer`, `--vectors`, and `--unsteered-only`
options are no longer needed or accepted.
