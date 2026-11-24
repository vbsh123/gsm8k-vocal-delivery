# Phi GSM8K bare-prompt evaluation

All **1,319** existing Jenny GSM8K questions, neutral and angry only: **2,638
answers**. There are no reserved/fitting questions, activation captures, vectors,
classifier selection, new TTS calls or steering. Happy and sad are not generated.

Exactly the same custom bare audio prompt as the DIFFA experiment:

```text
Solve the math problem in the audio. End your response with a final answer line in the format:
#### <number>
```

The runner first uses the existing
`outputs/emotion-block-night/phi/gsm8k/run_config.json`, or the corresponding
DIFFA night plan. Otherwise it resolves the existing Jenny audio inventory.
It validates both deliveries for every question and does not discard missing,
ambiguous or long recordings silently. `--source CONFIG` selects another complete
GSM night plan; `--audio-root DIRECTORY` relocates the same exact audio hashes.

Matching previously saved Phi bare answers in the original night folder are
copied with provenance when model/revision, prompt, seed, budget, native decoding
settings, audio hashes, reference, voice and absence of steering match. The old
experiment is read only. Own output resumes per completed answer. A conflicting
answer or changed recording/config is rejected.

Pinned `microsoft/Phi-4-multimodal-instruct` revision
`93f923e1a7727d1c4f446756212d9d3e8fcc5d81`, native audio processor, BF16, SDPA,
greedy generation, seed42,512 maximum new tokens. No instruction requesting or
forbidding reasoning. Existing native Phi generation is called with
`collect=False, interventions=[]`; no activation hook is installed. Shared GPU
lock and idle check are used. Per-answer logs show suffix token counts and time.

Summary uses the installed upstream GSM8K saved-answer filters and exact-match
metric; numeric equivalence and format extractability are additional diagnostics.
This is an audio prompting experiment, not the official full text-harness
inference protocol. Accuracy and mean/median tokens use identical completed
neutral/angry pairs. Exact paired McNemar p and paired bootstrap95% CI are printed;
all tests and interim summaries are exploratory and unadjusted.

On the Phi machine with its existing model and audio:

```bash
git pull --ff-only
nohup bash run_phi4_gsm_bare_eval.sh > outputs/phi4-gsm-bare-eval.log 2>&1 < /dev/null &
tail -f outputs/phi4-gsm-bare-eval.log
```

CPU summary while running:

```bash
bash run_phi4_gsm_bare_eval.sh --mode summary
```

The wrapper preflights/scorers with `.venv-diffa/bin/python`, generates with
`.venv-bbh-phi4/bin/python`, then scores with the first environment. `--mode
prepare` does no inference and imports no previous answers. Results are saved in
`outputs/phi4-gsm-bare-eval/`.
