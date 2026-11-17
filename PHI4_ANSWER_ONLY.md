# Phi-4 answer-only follow-up on completed GSM8K and BBH-three pairs

This queue stops the currently running **Phi-4 GSM8K queue**, freezes its actual
completed neutral/angry pairs, then tests an answer-only instruction on exactly
those recordings. BBH-three uses all completed pairs from its original-prompt run
as a second task. The number is read at launch; it is not hardcoded to543.

```bash
git pull --ff-only
nohup bash run_phi4_answer_only.sh > outputs/phi4-answer-only.log 2>&1 < /dev/null &
tail -f outputs/phi4-answer-only.log
```

If GSM8K was already stopped manually, the stop stage reports that and continues.
Stopping targets `run_gsm_candidate.sh phi4`, its worker, and their child
processes, only in this repository. It sends SIGTERM, never SIGKILL. It does not
stop BBH-five, BBH-three, Kimi, other models, or a log viewer. Completed answers
are retained; an interrupted in-flight answer can be absent. It checks that the
GSM queue has exited before freezing questions. A stopped worker may leave an
incomplete final JSONL fragment: the read-only snapshot omits it, without editing
the original file. Unpaired final answers are excluded from both prompt arms.

## Controlled change

The source is each task's **original** prompt. Replace ONLY:

> Let's think step by step.

with:

> Do not show reasoning or explanations. Give only the final answer.

Keep all other wording, input audio, model revision, seed42, greedy decoder,
repetition penalty1, and512-token budget unchanged. GSM keeps `#### <number>`;
BBH-three keeps `The answer is (<letter>).` No shortened token budget, waveform
changes, new TTS, checkpoint download, steering, or correctness-based selection.
The instructions affect **written reasoning**, not directly internal reasoning.

For each task the first launch freezes every question/voice with completed
neutral AND angry answers. It snapshots those baseline responses into the new
experiment, restricts both arms to the same questions, and records config/data
hashes. Answers that were correct, incorrect, formatted or unformatted are all
eligible. Resumes reuse the frozen cohort even if the original result file later
grows. This follow-up was chosen after examining interim results and is therefore
exploratory; stopping at an observed result is not a confirmatory sample design.

Only the answer-only arm requires inference: **2 × completed GSM pair count +
2 × completed BBH pair count** generations. The existing Phi-4 environment and
weights are reused. One model is loaded for both tasks. The queue waits for
already queued Phi-4 BBH-three/BBH-five jobs and then the shared GPU lock; it
doesn't interfere with another model's work.

## Files and results

Sources by default:

- `outputs/gsm8k-candidates/phi4/original/`
- `outputs/bbh-three-candidates/phi4/original/`

Follow-up outputs:

- `outputs/phi4-answer-only-followup/gsm8k/{original,answer-only}/`
- `outputs/phi4-answer-only-followup/bbh3/{original,answer-only}/`

Each task also has frozen `selection.json` and `comparison.json`. Raw source
answers/configs remain untouched. Preserve the original audio folders too.

```bash
.venv-diffa/bin/python phi4_answer_only.py --mode summary
```

Uses the same installed upstream GSM8K/BBH task filters and exact-match metrics,
with the same additional numeric and format diagnostics. Reports both prompt
arms, then within-delivery original-versus-answer-only comparisons on questions
complete in all four cells. `comparison.json` includes the change in the
angry-minus-neutral effect and its question-paired bootstrap95% interval, plus
response-token means/medians. Intervals/tests are unadjusted. A change in the
effect supports an instruction-dependent result, not proof that response length
itself caused the accuracy change. Neither prompt can guarantee zero internal
reasoning; no changes to hidden states are made.

Individual CPU preparation, without stopping a job or starting inference:

```bash
.venv-diffa/bin/python phi4_answer_only.py --mode prepare
```

Use `--gsm-run`, `--bbh-run`, and `--output` for explicit alternate locations.
The defaults compare against ORIGINAL prompts only, not the explicit-reasoning
arm. A different follow-up cohort needs a separate output directory.
