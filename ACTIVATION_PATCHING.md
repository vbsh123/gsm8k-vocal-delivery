# Matched-state audio activation patching

`activation_patch_audio.py` is a diagnostic that requires **both neutral and angry
recordings of every question**. It is not a single-recording inference method or
a learned steering vector. It leaves all existing vectors and results unchanged.

The default sample is 100 question/voice pairs, shuffled with seed 42 from the
two evaluation cohorts. Both recordings must fit 30 seconds. There is no emotion
classifier or correctness filter. The frozen sample and waveform fingerprints
are stored in `run_config.json`.

Each question gets fresh neutral and angry reference answers, then three
separate interventions: L12 acoustic, L12 answer, and L31 answer. On every
recipient forward, the script combines the angry audio/prompt embeddings with
the recipient's **current 512 answer embeddings**, including its committed
tokens and masks. It performs an additional donor forward, then replaces the
chosen recipient layer's token group with the donor's group. Answer positions
align by their common 512-token suffix; the 64 acoustic tokens align by the
insertion bounds. Variable-length semantic tokens are not patched.

This avoids replaying activations from an independent angry decoding trajectory.
The donor is recomputed for the current state. One model stays on the GPU;
there is no stored activation collection. Each patched generation costs about
two normal decoder passes per denoising forward. The 100-question run has 500
reference/patched generations, five additional identity-control generations,
and brief prefix-capture calls that stop before decoder computation.

L31 answer replacement is a **positive control**, because these are the final
decoder block's answer representations. It should reproduce unpatched angry
generation, not establish that L31 uniquely mediates emotion. L12 is the more
informative intervention. Acoustic self-patches on the first five sampled pairs
must reproduce neutral responses exactly; L31 positive controls must reproduce
angry responses exactly. A failed control stops the run before accepting that
patch's result.

Run on the original steering machine:

```bash
git pull origin main
nohup .venv-diffa/bin/python -u activation_patch_audio.py \
  > outputs/jenny-activation-patch100.log 2>&1 < /dev/null &
tail -f outputs/jenny-activation-patch100.log
```

It waits for existing steering/emotion workers, the shared GPU lock and active
GPU work. Completed results resume. The neutral and angry references are fresh
to support exact replay controls with the current model. Nothing is passed the
gold answer or question transcript during generation. Prompt and decoding
settings match the earlier GSM8K experiments; seed 42 resets each generation.

```bash
.venv-diffa/bin/python activation_patch_audio.py --summary-only
```

All comparisons are exploratory and paired with the same neutral recording.
To extend to all eligible pairs later, use `--num-questions 0` and a **new**
`--output`; changing the sample in an existing output is rejected.
