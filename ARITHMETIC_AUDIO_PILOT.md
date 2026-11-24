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
