# SAME100 instruction reference directions

## Polite format wording extension (only 100 new forwards)

After the original ten controls have all100 states, run:

```bash
git pull --ff-only
nohup bash run_instruction_controls100.sh diffa --extend-format-please \
  > outputs/diffa-format-please100.log 2>&1 < /dev/null &
tail -f outputs/diffa-format-please100.log
```

The replacement sentence is: "Please make sure your final answer follows the
requested format exactly." It replaces the reasoning sentence as before;
the neutral audio,100 questions, seed, task and final-answer suffix are unchanged.
The extension freezes hash receipts for all reused states, leaves the original
control config/states/summaries untouched, and collects only100 new forwards
plus the existing one-forward reference replay. New state size is about200MiB
for DIFFA,80MiB for Phi. Use `phi4` instead of `diffa` for that model.

New output: `outputs/diffa-instruction-controls100-format-please/`. CPU inspection:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python -u collect_instruction_controls.py \
  --model diffa --extend-format-please --mode summary
```

It reports all old and new directions against answer-only and bare-task states.
Two additional references, `polite` and `format`, compare only the new condition
against those earlier conditions. These are conditional wording contrasts,
not isolated effects of the word "please" or proof of additive mechanisms:
the new sentence changes more than one word. The extension was selected after
inspecting original controls and is exploratory.

## Original ten-control experiment

Use the exact question/voice pairs, full neutral audio, parent prompts and token
capture sites from the existing DIFFA or Phi reasoning/emotion collector.
There is no new selection, TTS, steering, accuracy scoring or full generation.

```bash
git pull --ff-only
nohup bash run_instruction_controls100.sh diffa \
  > outputs/diffa-instruction-controls100.log 2>&1 < /dev/null &
tail -f outputs/diffa-instruction-controls100.log
```

For Phi on its machine, replace `diffa` with `phi4` in both script argument and
log name. Uses the existing `.venv-bbh-phi4` and pinned model. DIFFA uses the
existing `.venv-diffa` and shared model loader. Both wait for active queues and
the shared GPU lock, and resume per question/instruction. Keep their parent
SAME100 state folders intact. CPU summary command:

```bash
.venv-diffa/bin/python collect_instruction_controls.py --model diffa --mode summary
.venv-diffa/bin/python collect_instruction_controls.py --model phi4 --mode summary
```

`--mode summary` reads completed state files without taking the collection
lock or writing configs/summaries, so it can inspect an active collection.
The collection process writes the final CSV/JSON after completing its CPU
summary. GPU memory can remain reserved during this CPU calculation.

Defaults:

- DIFFA parent `outputs/diffa-reasoning-emotion100`, output `outputs/diffa-instruction-controls100`.
- Phi parent `outputs/phi4-reasoning-emotion100-gsm8k`, output `outputs/phi4-instruction-controls100`.
- `--parent` / `--output` override folders; intended for these GSM8K parents.

Each model adds 1,000 first forwards: ten neutral-audio instruction variants on
the same 100 questions. Ten variants are two reasoning paraphrases, brevity,
elaboration, format adherence, checking calculations, attention to detail,
confidence, politeness, and omission of the extra instruction sentence.
One original neutral-reasoning forward is also replayed on each model load to
check that the current runtime reproduces the saved reference (1e-4 absolute
and relative tolerance). A mismatch stops collection rather than mixing model
states; `reference-replay.json` records the result. This is a sanity check on
one question, not proof of runtime equivalence on all questions.

Only the reasoning sentence changes; the task wording and final-answer suffix
stay fixed. Reasoning and emotion parent states are reused, not recomputed.
Approximate new state storage: 2 GiB for DIFFA and 0.8 GiB for Phi.

Primary comparisons use the original active answer-only reference:

`direction(instruction) = mean(neutral_instruction - neutral_answer_only)`

`emotion = mean(angry_reasoning - neutral_reasoning)`

Thus the `original_reasoning` row reproduces the original comparison. Since
answer-only explicitly prohibits explanation, all directions are also reported
against the bare task (no extra instruction sentence). The emotion vector stays
unchanged. This distinguishes choice of reference from choosing an instruction.

Outputs: frozen config including parent hashes, runtime, resumable new states,
`instruction-cosines.csv`, and `summary.json`. Rows contain all layers/groups,
signed cosine, signed projection coefficient, squared-norm overlap, norms, and
both cross-half instruction/emotion cosines. Terminal output highlights answer
L26 and L31 for every instruction; these are previously inspected layers, not
newly selected evidence. Incomplete conditions are omitted from summary until
all100 states exist; no table silently changes its cohort size.

This is a calibration of instruction-request geometry. It does not prove
reasoning, behavioral causation, or statistical specificity. Some controls can
also induce reasoning. Prompt length/style/position effects remain, and models
still have different native capture sites and potentially different question
cohorts. Cross-half checks do not remove common prompt effects. Compare the
whole layer pattern and report all chosen controls rather than only a winning
instruction or layer. No similarity p-values are claimed.
