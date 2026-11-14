# SVAMP: 500-question frozen transfer experiment

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

Steering is disabled by default. Only explicit `--steer` adds the same neutral audio
with frozen GSM8K L31 answer steering at strength 1.

No SVAMP vector extraction or tuning. When `--steer` is explicitly requested, the original mean checkpoint is discovered
from the existing GSM8K winner config or original vector folders, or supplied with
`--vectors`. It is fingerprinted in the run configuration. Do not substitute the
CommonsenseQA, SVD, forward-window or reversed/random checkpoints.

DIFFA uses the existing shared loader/generator, the identical GSM8K math prompt,
512 output tokens, 128 steps, 32-token blocks, fast-dLLM, temperature zero, and
seed 42 reset before each condition. Only one model is loaded. CPU/cloud TTS runs
first; inference waits for the shared GPU lock and existing GPU processes.

The primary score is **flexible numeric correctness**: the upstream GSM8K harness
extractor, followed by our numeric-equivalence comparison. The report also includes
upstream-style string matching and our existing explicit-final-answer strict score.
This is not an official SVAMP harness inference task. Saved responses are scored
without using gold answers to choose the extracted prediction. All comparisons
against neutral include paired McNemar p-values and paired bootstrap confidence
intervals; these are unadjusted. Original response files remain untouched.

On the steering machine, with Azure environment variables exported:

```bash
git pull --ff-only
mkdir -p outputs
nohup .venv-diffa/bin/python -u run_svamp500.py \
  > outputs/svamp500.log 2>&1 < /dev/null &
tail -f outputs/svamp500.log
```

Run only generation with `--mode generate`, or only evaluation with `--mode
evaluate`. Neutral/angry accuracy is the default. Rerunning resumes
saved audio and answers. Configuration changes require a separate output folder.

Check completed results at any time, without GPU work:

```bash
.venv-diffa/bin/python -u run_svamp500.py --mode summary
```

Recordings and a listening page: `outputs/azure-svamp500/`.
Raw answers, frozen config, upstream scorer provenance, rescored answers and
summaries: `outputs/svamp500-diffa2/`.

An earlier three-condition run can resume without `--steer`: only pending neutral/angry
answers are generated. Its original config and all raw answers are preserved; the
default summary excludes steering.
