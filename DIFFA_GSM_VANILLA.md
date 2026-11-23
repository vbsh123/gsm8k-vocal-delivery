# Vanilla GSM delivery evaluation and fitting

This experiment evaluates neutral, angry, happy and sad Jenny audio with exactly:

```text
Solve the math problem in the audio. End your response with a final answer line in the format:
#### <number>
```

Collection and evaluation use the same prompt and decoding recipe: seed42,
512 new tokens, 128 diffusion steps, block32, temperature0. After natural
delivery evaluation and fitting finish, each vector steers the same fixed500
neutral deliveries at L31/answer x+1. Raw mean vectors retain their original
magnitudes. No audio generation or classifier inference occurs.

From the complete1319-question four-delivery pool, sort IDs, shuffle with seed42
and reserve the first500 as a common steering holdout **before consulting emotion labels**.
Natural delivery accuracy is evaluated on **all1319 questions**, with that
fixed500 subset also reported separately. Including fitting questions in natural
delivery evaluation does not apply a fitted vector to them.
All fitting excludes these500. Sample425 fitting IDs with seed43 for the random
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

The fixed500 natural deliveries run first. Collection then traverses the union of
selected fitting questions. Overlapping regimes reuse captures, and selected
emotions reuse the neutral capture on overlapping IDs. Running aggregates retain
raw responses and drift summaries, with atomic per-regime checkpoints and no full
per-question activation cache. Repeating the launch safely resumes completed work.
Collection responses are reused as natural deliveries with exact audio/prompt/
recipe validation, then the remaining deliveries complete the full1319 pool.
Then each selected emotion runs two steering evaluations: random-vector x+1 and
agreed-vector x+1, both on the common500 holdout. Zero-strength baseline answers
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

Artifacts live in `outputs/diffa-gsm-vanilla`: `split.json`, `delivery/results.jsonl`,
and `<emotion>/{random,agreed}/style-vectors.pt`. `<emotion>/delivery-report`
compares all1319 natural deliveries, and `<emotion>/heldout-delivery-report`
compares the fixed500 subset; partial reports explicitly show the matched count.
Each `<emotion>/<regime>/strength-1` contains held-out x+1 steering answers,
configuration, scored rows and accuracy/token-length summaries. The same summary
command shows these as they run. Cached rows must match vector hashes,
interventions, prompts, audio and decoding settings to be reused.
Collection reports are descriptive fitting results, not held-out evidence.
Tests are exploratory and unadjusted. Direct emotion comparisons require the raw
delivery rows from both machines to be brought together later; they cannot be
inferred from which neutral comparisons reach significance.

This is a new split and prompt experiment. Compare random versus agreed vectors
on this frozen evaluation set later, not against historical gains on other cohorts.
