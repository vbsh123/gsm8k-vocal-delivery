# IFBench after IFEval: neutral, angry, happy

This is a spoken adaptation of the official IFBench **300 test prompts**, using
the released verification code at revision `488918665d53f867fa908a293769ec6e71e74aec`.
It is not the IF-RLVR training set or the optional multiturn dataset.

The default prepares **900 Azure clips**: Jenny, neutral/angry/happy (Azure
cheerful), intensity 1.5. Prompt wording is unchanged between styles and the
original constraint metadata is retained for the verifier. No emotion labels,
transcript, custom answer marker or constraint metadata are supplied separately
to DIFFA. The text wrapper is only “Follow the instructions in the audio.”
No steering, vector extraction, or parameter search.

The queued shell launcher first waits for IFEval Python/shell jobs in this repo to
exit, then installs the official verifier's small dependencies, downloads the
needed NLTK resources, generates audio and evaluates. It does not change the
environment while IFEval is running. Inference also waits for the shared GPU lock
and any active GPU process. An exited failed IFEval job also releases the queue;
inspect its log separately to distinguish failure from completion.

The same native DIFFA input rule as IFEval applies: every recording must fit 30
seconds in all **three** requested styles. No cropping or stretching. Longer
questions remain saved but are excluded from all three comparison conditions;
`eligibility.json` lists durations and exclusions. Therefore the actual evaluation
cohort can be smaller than 300. Punctuation, symbols and exact-string constraints
may be imperfectly expressed by TTS; listen to examples and report this limitation.
We do not silently apply the experimental BBH long-audio adaptation here.

DIFFA generation uses seed 42 reset per recording, fast-dLLM, 128 steps, 32-token
blocks, temperature zero and 1024 output tokens, matching the current IFEval run.
Output length can limit some constraints; token counts and near-budget response
counts are reported. Weights are loaded once and existing HF cache settings are
reused.

Evaluation calls **unmodified upstream IFBench strict and loose checkers**.
Loose prompt-level accuracy is primary, following IFBench's reported convention.
Strict prompt-level and both instruction-level scores are also shown, with paired
McNemar tests and 200,000-resample bootstrap intervals. Tests/intervals and any
constraint breakdowns are unadjusted. This is not a factual-answer accuracy score.
Every selected verifier is exercised before spending Azure or GPU time.

From the IFEval machine, with Azure key and region already exported:

```bash
git pull --ff-only
mkdir -p outputs
nohup bash run_ifbench_after_ifeval.sh \
  > outputs/ifbench-after-ifeval.log 2>&1 < /dev/null &
tail -f outputs/ifbench-after-ifeval.log
```

Check saved results anytime, CPU only:

```bash
.venv-diffa/bin/python -u run_spoken_ifbench.py --mode summary
```

Audio, metadata and listening page: `outputs/azure-spoken-ifbench/`.
Configuration, raw responses, official scores and summary:
`outputs/spoken-ifbench-diffa2/`.
Upstream source and license: `external/ifbench/`, excluded from Git.

Rerun the same launcher to resume existing clips/answers. Changed prompts, audio,
verifier code or generation settings require a separate result folder.
