# Phi BBH5 and MultiArith bare-task pilots

**100 fixed questions per dataset**, each with neutral and angry audio:200
answers per dataset,400 total. Use the first100 of the seed42 shuffled complete
BBH5 (250) and MultiArith (580) question banks. Keep this selection on resume.
No outcome, duration or emotion-classifier filtering. No new TTS, activation
collection, vectors or steering; happy and sad are not generated.

These are exactly the **custom bare audio prompts** used in the DIFFA night:

BBH5:

```text
Solve the problem in the audio. End your response with a final answer line in the format:
The answer is (<letter>).
Replace <letter> with A, B, C, D, or E.
```

MultiArith:

```text
Solve the math problem in the audio. End your response with a final answer line in the format:
#### <number>
```

Existing complete night configs under `outputs/emotion-block-night/{phi,diffa}`
are preferred. Otherwise resolve the already extracted Jenny ZIP inventory.
Each selected question needs both exact recordings and unambiguous source text
and gold answer. BBH5's original input/target fields are recovered through the
same checked scoring-bank helper as DIFFA. `--audio-root DIRECTORY` selects the
existing extracted audio bundle. `--source CONFIG` can select one dataset's
complete night config when `--datasets` contains only that dataset.

Use pinned Phi-4 revision `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`, native
full-audio processor, BF16, SDPA, seed42, greedy decoding and512 maximum new
tokens. Capture is disabled and the intervention list is empty. The queue waits
for the shared GPU lock used by GSM, loads Phi once, then runs BBH5 and MultiArith
sequentially. It does not stop the current GSM job. Every completed answer is
saved and logs its suffix token count and elapsed time. Rerunning resumes.

Scoring uses the same saved-answer evaluators as DIFFA: BBH5's upstream
strict/flexible filters and exact match; MultiArith's GSM8K strict/flexible
filters plus numeric-equivalence and format-extractability diagnostics. The
generation prompts are audio adaptations, not full official text-harness
protocols. Summaries use matched neutral/angry IDs for accuracy and mean/median
token counts, with exact paired McNemar p and paired bootstrap95% CI. Tests are
unadjusted and exploratory. A CPU summary runs after each dataset while the Phi
process retains the model for the next one.

On the Phi machine:

```bash
git pull --ff-only
nohup bash run_phi4_more_bare_eval.sh > outputs/phi4-bbh5-multiarith-bare100.log 2>&1 < /dev/null &
tail -f outputs/phi4-bbh5-multiarith-bare100.log
```

Check both or either dataset:

```bash
bash run_phi4_more_bare_eval.sh --mode summary
bash run_phi4_more_bare_eval.sh --mode summary --datasets bbh5
bash run_phi4_more_bare_eval.sh --mode summary --datasets multiarith
```

Results: `outputs/phi4-bare-eval100/bbh5/` and
`outputs/phi4-bare-eval100/multiarith/`. These are separate from the GSM run.
`--mode prepare` validates inputs and evaluator availability without inference.
The wrapper uses `.venv-diffa` for scoring and `.venv-bbh-phi4` for inference.
