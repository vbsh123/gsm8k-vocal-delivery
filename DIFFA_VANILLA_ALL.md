# Next datasets: vanilla generation, two vectors, x+1 steering

The default queue covers SVAMP, MultiArith, BBH3, BBH5 and IFEval sequentially.
GSM keeps its existing425-question vector selections in `outputs/diffa-gsm-vanilla`.
It can optionally be included with `--datasets gsm8k ...`; it is never resampled.

Every natural question receives neutral and the requested emotional delivery.
Activation capture happens during that same generation pass on fitting questions.
There is no separate500-question generation limit and no shared3 filter.
No TTS or classifier inference is launched: existing Jenny audio and saved e2v
labels are verified before loading the model.

Each emotion gets two vectors:

- Random: half the total questions, sampled without filtering emotion labels.
  The same random half is used for angry, happy and sad.
- Agreed: independently sampled from clips recognized as the requested emotion,
  using as many as available, capped at half the total. Neutral predictions are
  ignored. The two regimes may overlap. No correctness filtering is used.

With the audited audio/label bundle:

| Dataset | All natural questions | Random per emotion | Agreed angry/happy/sad |
|---|---:|---:|---:|
| SVAMP | 1000 | 500 | 500 / 500 / 500 |
| MultiArith | 580 | 290 | 290 / 290 / 290 |
| BBH3 | 250 | 125 | 125 / 125 / 123 |
| BBH5 | 250 | 125 | 125 / 125 / 98 |
| IFEval | 541 | 270 | 270 / 270 / 270 |

Question traversal uses seed42, random-half selection seed43, and agreed
selection seed44. All selections are frozen before inference. Both machines
print a portable fingerprint covering questions, golds, audio hashes, predictions,
prompts, decoding settings and selections; fingerprints must match.

Math tasks use512 new tokens and exactly:

```text
Solve the math problem in the audio. End your response with a final answer line in the format:
#### <number>
```

BBH uses512 new tokens and:

```text
Solve the problem in the audio. End your response with a final answer line in the format:
The answer is (<letter>).
Replace <letter> with A, B, or C.
```

BBH5 substitutes `A, B, C, D, or E` on the final line. Neither prompt requests
reasoning. Official BBH golds, inputs and scoring filters are retained.

IFEval preserves its approved bare prompt and1024-token budget:

```text
Follow the instructions in the audio.
```

Every task uses seed42,128 diffusion steps, block32 and temperature0. At the end
of each dataset, both raw mean vectors steer L31/answer at x+1 on **every neutral
question outside that vector's own fitting set**. Unsteered baselines come from
the natural generation pass. Different fitting sets can yield different steering
questions; direct vector comparisons must use questions held out from both.
All natural emotion comparisons use the same full question pool.

The commands below wait for the shared DIFFA GPU lock and an idle GPU, so they
can queue behind the running GSM job. The model is loaded once for the sequence.
Completed responses and aggregate checkpoints resume safely after interruption.
The queue does not stop any process.

Original DIFFA machine, angry:

```bash
git pull --ff-only
nohup bash run_diffa_vanilla_all.sh --emotions angry >> outputs/diffa-vanilla-angry.log 2>&1 </dev/null &
```

New DIFFA machine, happy and sad:

```bash
git pull --ff-only
nohup bash run_diffa_vanilla_all.sh --emotions happy sad >> outputs/diffa-vanilla-happy-sad.log 2>&1 </dev/null &
```

To prepare and verify inputs without generation:

```bash
bash run_diffa_vanilla_all.sh --mode prepare
```

Progress and accuracy:

```bash
bash run_diffa_vanilla_all.sh --mode summary --emotions angry
bash run_diffa_vanilla_all.sh --mode summary --emotions happy sad
```

Use `--datasets svamp multiarith` to limit the queue or report. Add `--detailed`
only when fitting-set accuracy diagnostics are needed. Default summaries show
full-pool natural accuracy, vector collection progress, and x+1 steering when
available. Artifacts are under `outputs/diffa-vanilla/<dataset>`. Fitting answers
are retained for descriptive analysis; held-out steering results never include
their own vector's fitting IDs. Tests and paired intervals are exploratory and
unadjusted.
