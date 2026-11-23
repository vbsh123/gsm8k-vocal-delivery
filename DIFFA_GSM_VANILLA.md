# Vanilla GSM delivery evaluation and fitting

This experiment evaluates neutral, angry, happy and sad Jenny audio with exactly:

```text
Solve the math problem in the audio. End your response with a final answer line in the format:
#### <number>
```

Collection and evaluation use the same prompt and decoding recipe: seed42,
512 new tokens, 128 diffusion steps, block32, temperature0. After natural
delivery evaluation and fitting finish, each vector steers every neutral question
outside its own425 fitting IDs (894 questions) at L31/answer x+1. Raw mean vectors retain their original
magnitudes. No audio generation or classifier inference occurs.

Natural delivery accuracy is evaluated on **all1319 questions**, with activation
capture during that same generation pass on selected fitting questions.
The existing425 selections are frozen to preserve collected work. The first
implementation sampled them from819 questions after a seed42 shuffle; its old
500-question split remains in metadata only and no longer restricts generation
or steering. Sample425 fitting IDs with seed43 for the random
regime, shared across emotions. Independently sample425 recognized fitting IDs
per emotion with seed44 for the agreed regime. Agreement concerns only the
requested emotion; neutral predictions are ignored. The agreed cohorts may
overlap but need not be identical. No response-accuracy filtering is used.

Both machines must restore the complete same Jenny audio ZIP and saved emotion2vec
results. Every launch prepares all three selections and prints a portable split
SHA256, covering IDs, golds, question text, audio hashes, labels, prompt and recipe.
Production launches require the audited fingerprint
`e02dabe85aacd64ab4b9595959aed518eac33cde602bf54eeec3253ab8755873` on both
machines; mismatches fail before inference. All manifests are immutable; if
inputs change, preparation fails rather than silently reusing an incompatible run.

Original DIFFA machine:

```bash
git pull --ff-only
nohup bash run_diffa_gsm_vanilla.sh --emotions angry >> outputs/diffa-gsm-vanilla-angry.log 2>&1 </dev/null &
```

New DIFFA machine:

```bash
git pull --ff-only
nohup bash run_diffa_gsm_vanilla.sh --emotions happy sad >> outputs/diffa-gsm-vanilla-happy-sad.log 2>&1 </dev/null &
```

Generation and activation collection now happen together, question by question.
Overlapping regimes reuse captures, and selected emotions reuse the neutral
capture on overlapping IDs. Running aggregates retain
raw responses and drift summaries, with atomic per-regime checkpoints and no full
per-question activation cache. Repeating the launch safely resumes completed work.
Existing collection responses are reused as natural deliveries with exact
audio/prompt/recipe validation; existing completed aggregates are not recollected.
Old saved text without activations cannot reconstruct an unfinished fitting pair:
only those missing activation captures must be rerun.
Then each selected emotion runs two steering evaluations: random-vector x+1 and
agreed-vector x+1, each on all894 questions outside that vector's425 fitting IDs.
Direct comparisons between vectors must use the intersection held out from both,
since their fitting sets differ. Zero-strength baseline answers
are copied from the exact natural neutral responses; they are not regenerated.
The full natural-delivery manifest is a separate `delivery/run_config.json`;
old fitting configs, split hashes, checkpoint prefixes and completed500 answers
are preserved. Rerun the same commands after pulling to extend an earlier run.
If a vanilla runner is still active, the new launch waits for it to finish.
The runner also waits for the shared DIFFA lock and an idle GPU before loading weights.
It never stops any existing process.

CPU reports while running, using the corresponding emotion selection:

```bash
bash run_diffa_gsm_vanilla.sh --mode summary --emotions angry
bash run_diffa_gsm_vanilla.sh --mode summary --emotions happy sad
```

Summaries also print saved emotion2vec agreement for each emotion across all
clips and on the exact completed neutral/emotion accuracy pairs. Counts come
from the frozen exact-audio label snapshot, with no new classification. These
percentages do not filter accuracy. `emotion2vec-agreement.json` additionally
records agreement within each vector's fitting selection.

For **accuracy restricted to recognized emotional recordings**, use:

```bash
bash run_diffa_gsm_vanilla.sh --mode summary --emotions happy sad --agreed-only
```

This requires recognized happy AND sad on the same questions, plus completed
neutral/happy/sad responses. It prints paired accuracy and direct happy/sad tests
on that one common cohort, not just classifier agreement percentages. Neutral
recognition is not required. For one emotion, request just `--emotions angry`
or `--emotions happy` to use that emotion's recognized subset.

To compare angry on the original machine against happy/sad on the new one with
the same classifier selection, add `--agreement-emotions angry happy sad` to
both commands. All three labels are available in the frozen metadata even when
responses are split across machines. Partial completion can still produce
different matched Ns until both runs finish.

Filtered results are saved separately under `agreed-accuracy/<required-emotions>`.
The options are CPU-only summaries and never change generation or vector fitting.

Artifacts live in `outputs/diffa-gsm-vanilla`: `split.json`, `delivery/results.jsonl`,
and `<emotion>/{random,agreed}/style-vectors.pt`. `<emotion>/delivery-report`
compares all1319 natural deliveries; partial reports show the matched count.
The default summary prints natural accuracy, a one-line progress count per vector,
and steering accuracy when available. Add `--detailed` for fitting-set accuracy;
it is not a separate evaluation experiment. No fixed500 accuracy block is printed.
Each `<emotion>/<regime>/strength-1-unfitted` contains held-out x+1 steering answers,
configuration, scored rows and accuracy/token-length summaries. The same summary
command shows these as they run. Cached rows must match vector hashes,
interventions, prompts, audio and decoding settings to be reused.
Collection reports are descriptive fitting results, not held-out evidence.
Tests are exploratory and unadjusted. Direct emotion comparisons require the raw
delivery rows from both machines to be brought together later; they cannot be
inferred from which neutral comparisons reach significance.

This is a new prompt experiment. Compare vectors on common questions held out
from both fits, not against historical gains on other cohorts. Earlier500-only
steering artifacts, if present, are preserved in the old `strength-1` folders.
