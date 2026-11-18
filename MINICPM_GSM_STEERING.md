MiniCPM uses the pinned native model/processor, BF16 SDPA, seed42, greedy512-token
generation and the same reasoning prompt/native chat options as its SAME100
diagnostic. No transcript, gold answer or emotion text is passed to the model.

The canonical1269 question bank is shuffled with seed42: first400 neutral/angry
pairs collect the raw L30 answer-position difference; remaining869 neutral clips
are each evaluated without steering and with +1. Collection and evaluation share
the same hook position: last prediction position on every native decoder forward,
including prefill. Per-generation means are differenced per question, then
averaged equally over400 questions. This is not the initial-forward cosine vector.
The earlier100 diagnostic questions may overlap test questions; this experiment
is exploratory. No outcome/classifier/duration filtering or audio cropping.

Run both models:

```bash
nohup bash run_phi_minicpm_gsm_parallel.sh > outputs/phi-minicpm-gsm-parallel.log 2>&1 </dev/null &
```

Phi first waits for its existing follow-up queue and shared lock; its extra-stage
concurrent flag allows MiniCPM and caps Torch at20GiB. The coordinator starts
MiniCPM after Phi GPU initialization, or after Phi exits if already completed.
MiniCPM skips the shared GPU lock, requires24GiB free at startup and caps Torch
at22GiB. These caps do not cover all non-Torch allocations and cannot guarantee
peak-memory compatibility. Neither script stops another job or deletes data.
Do not launch a duplicate Phi -2 job if it is already queued/running; use the
individual MiniCPM wrapper after Phi starts in that case.

Outputs: `outputs/minicpm-gsm400-869-L30-a1`: frozen config/split, per-pair CPU
means with collection responses, style vector, all1738 test responses, runtime,
rescored responses, upstream harness provenance and paired summaries.
Atomic collection caches and per-answer output support resuming.

CPU progress:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python steer_minicpm_gsm.py --mode summary
```

Scoring uses the installed upstream GSM8K filters and exact-match metric on saved
audio-generated responses, with numeric equivalence as an additional metric;
this is an audio adaptation, not full official text-harness inference.
Format extraction is a separate diagnostic. Paired tests/intervals are unadjusted.

Evidence backup includes this run with `pack_experiment_backup.py --only phi minicpm`.
