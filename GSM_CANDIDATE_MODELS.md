# Phi-4 / candidate GSM8K delivery evaluation

Evaluate **all 1,269** questions from the three main Azure cohorts, in neutral and
angry. No collection, steering, classifier/outcome/duration filter, or question
subsampling. Uses the same full audio already generated for Kimi GSM8K at
`outputs/azure-kimi-gsm8k1269/`. Existing Kimi answers, splits and vectors stay in
their own folders and are never reused as candidate answers.

```bash
git pull --ff-only
nohup bash run_gsm_candidate.sh phi4 > outputs/gsm8k-phi4.log 2>&1 < /dev/null &
tail -f outputs/gsm8k-phi4.log
```

The default evaluates both prompt variants, separately, for **5,076 generations**:

- Original: "Let's think step by step."
- Explicit: "Give concise step-by-step reasoning with only the necessary calculations."

Only that sentence changes. Both keep the same math task instructions and final
`#### <number>` requirement. The explicit variant is the aligned prompt used for
the latest DIFFA/Kimi GSM8K runs. Both use seed42, native greedy decoding, one
beam, 512 new tokens and the same pinned Phi-4 snapshot and adapters as BBH.
Use `--prompt explicit` or `--prompt original` to select only one (2,538 answers).

The wrapper waits for the candidate's BBH queue, current Kimi queues, then the
shared GPU lock. It reuses the existing Phi-4 environment and cached checkpoint.
Missing audio can be resumed using the original Azure cohorts; only missing
clips require `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION`. Existing complete
audio requires no TTS requests. Nothing is cropped, and Phi-4's native processor
receives full 16kHz mono audio. Its card's40-second recommendation remains
documented in the BBH adapter notes; durations are retained in the run config.

```bash
.venv-diffa/bin/python evaluate_gsm_candidates.py --model phi4 --mode summary
```

Results live under `outputs/gsm8k-candidates/phi4/{original,explicit}/`: raw
answers, frozen config, runtime/package versions, hashes, exact suffix token
counts, upstream scorer provenance, rescored answers and summary. Rerunning
resumes each missing question/style separately. Prompt variants are never pooled.

Primary metric is numeric correctness using installed upstream GSM8K flexible
extraction plus numeric equivalence. Strict and flexible upstream exact match,
format compliance, mean/median tokens, exact paired McNemar tests and95% paired
bootstrap intervals are also reported. Comparisons always use identical matched
questions. This is audio-adapted inference with upstream scoring, not full
official text-only harness prompting. P-values and intervals are unadjusted.

The same runner supports `minicpm` and `flamingo` later, using their own isolated
environments and checkpoint pins. No additional model is downloaded for a Phi-4
run. Local CPU tests cover all-question selection, audio reuse, prompt parity and
actual installed harness scoring; GPU execution remains on the user's machine.
