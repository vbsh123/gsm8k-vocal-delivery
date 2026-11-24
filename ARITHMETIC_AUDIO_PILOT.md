# BBH arithmetic: 50-question audio pilot

Run `bash run_arithmetic_audio_pilot.sh` on a DIFFA machine with the usual Azure
Speech key/region exported. Requires the existing `.venv-diffa` and model cache.
Use `--mode summary` for CPU-only accuracy, token means/medians and matched
comparisons with exact McNemar p-values and paired bootstrap 95% intervals.
`--mode prepare` prepares requests and checks the evaluator without paid TTS or inference.
Rerunning the same command resumes completed clips and answers. The GPU phase
waits for the existing shared lock and for an idle GPU.

The sample is the first 50 questions after a seed42 shuffle of the authors' 250
`multistep_arithmetic_two` questions at revision
`9ee07bd481feebf959a6b59d61ea57bdcf30964d`. Every target is checked by a restricted
integer-expression interpreter before selection. There is no outcome filtering.

We synthesize 100 Jenny clips (neutral and angry for each question) and generate
200 answers: neutral bare, angry bare, neutral supply reasoning, angry supply
reasoning. Each recording is identical in its bare and reasoning conditions.
No happy/sad, demonstrations, fitting,
emotion classification or steering. Native DIFFA decoding uses the existing
recipe, seed42, temperature0 and a 512-token response budget for every condition.

Bare prompt:

```text
Answer the question in the audio. End your response with a final answer line in the format:
The answer is <number>.
```

Reasoning prompt (neutral and angry):

```text
Answer the question in the audio. Supply reasoning before the final answer. End your response with a final answer line in the format:
The answer is <number>.
```

These are custom audio generation prompts. The installed harness's
`bbh_cot_zeroshot_multistep_arithmetic_two` task provides saved-answer extraction
and exact-match scoring; its text generation prompt is not sent to DIFFA.

## Spoken representation

Every opening/closing parenthesis is spoken explicitly as "open parenthesis" /
"close parenthesis". A unary minus is "negative" and subtraction is "minus";
addition is "plus" and multiplication is "times". Operands are English number
words. The equals sign is replaced by a question frame, "What is the value of:
... ?". Arithmetic precedence and grouping are preserved. The fixed source uses
only single-digit operands and these operators. Unknown syntax is rejected.
Original notation, original target, spoken question, source index and stable ID
are saved together, avoiding ambiguity between source IDs and WAV batch numbers.

Audio, source and `listen.html` are under `outputs/azure-bbh-arithmetic50/`.
Answers, frozen prompts/configuration, decoder/model receipts and summaries are
under `outputs/diffa-bbh-arithmetic50/`. Other experiments are untouched.

Inspect three randomly sampled questions side by side in all four conditions
(12 saved responses, no scoring or generation):

```bash
python3 show_arithmetic_audio_outputs.py --count 3
```

The preview prints original expressions, the actual spoken questions, gold
answers, token counts and exact WAV paths. Selection uses completed matched
questions without filtering by accuracy or length. Use `--question-id
bbh_arithmetic_044` to inspect a specific source ID, including pending answers.

## Bare angry-minus-neutral steering pilot

```bash
bash run_arithmetic_steering.sh
bash run_arithmetic_steering.sh --mode summary
```

This uses the existing 50-question pilot and its audio, with a seed44 random
25-question fitting set and the remaining25 held out for steering. Neither
accuracy nor E2V labels select fitting questions. The pilot has no E2V labels.
The bare prompt is used for both neutral and angry fitting generations and all
steered evaluations. No Supply reasoning responses enter the vector or baseline.

Fresh fitting forwards capture current32-token answer-block means, matching the
existing block-collection method: matched chronological forward prefixes within
each pair, then equal weight per question. Saved bare responses are checked
against the fresh capture before committing each pair. Collection checkpoints
resume per completed pair, save layer-level aggregates and drift diagnostics,
and leave the original pilot intact.

The intervention is the raw L31 answer mean angry-minus-neutral vector at x+2,
applied to the answer buffer on the25 disjoint neutral evaluation recordings.
The existing neutral answers are reused after a neutral replay check. No TTS,
classifier, normalization, demonstrations or new prompt is introduced. Reporting
uses the BBH arithmetic task's own upstream filters, not GSM8K filters.

Output: `outputs/diffa-bbh-arithmetic-steering/`, with `collection/` holding the
checkpoint and vector and `L31-answer-a2/` holding paired responses and scores.
`--mode prepare` freezes metadata and checks scoring without GPU inference.
