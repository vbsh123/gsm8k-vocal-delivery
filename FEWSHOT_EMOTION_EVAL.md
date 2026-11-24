# DIFFA few-shot emotion evaluation

This condition uses published worked text examples and an **audio target question**.
It is an audio adaptation, not an official text benchmark replication. Evaluate all
neutral, angry, happy and sad Jenny recordings. No vector fitting, steering,
emotion2vec filtering, outcome filtering or reserved evaluation subset.

Sources are pinned by commit and exact SHA256 in
`data/fewshot_prompt_sources.json`; downloaded files remain under the new output
directory. Existing results, cohorts and vectors are never modified or reused as
responses in this condition.

## Prompt construction

BBH3 and BBH5 retain the task description and all three worked examples from the
authors' `cot-prompts/logical_deduction_*_objects.txt` files. The canary header and
delimiter are removed. **Both published files use the same three-object examples**;
BBH5's examples are preserved as published. Append:

```text
Q: The question and answer choices are provided in the audio.
A: Let's think step by step.
```

GSM8K retains all eight hardcoded question/target demonstrations from the pinned
harness `gsm8k-cot.yaml`, rendered as `Q: {question}\nA: {target}`, separated by
blank lines. Demonstrated answers end with `The answer is <number>.` Append:

```text
Q: Solve the math problem in the audio.
A:
```

The target question, choices and gold answer are never copied into the text
prompt. Demonstration/evaluation question overlaps cause preparation to fail.
Exact prompts are saved as `<output>/<dataset>/prompt.txt` and their sources,
hashes and adaptation in `prompt-provenance.json`.

## Generation and scoring

Use every existing night question/audio hash and the same order/decoding recipe
when a night cohort exists. On a machine without that dataset's night cohort,
resolve every complete four-style question from the extracted Jenny ZIP: 250
BBH3, 250 BBH5 or 1319 GSM8K. No labels are needed. Missing or ambiguous audio
causes failure rather than silently reducing the evaluation pool.

Native DIFFA chat wrapping inserts the audio before the text demonstrations.
This modality/position adaptation is part of the experimental condition. Native
DIFFA generation remains seed42, fast_dllm, 128 steps, 32-token blocks, temperature0,
512 generated-token budget. Harness stop strings are not applied, and saved
responses are not retrospectively truncated. The exact native text wrapper and
token count are recorded before generation; abort if it would exceed the native
4096-token text truncation limit. Model revision receipts and package versions
are frozen on first generation and checked on resume.

BBH saved responses use the same harness zero-shot CoT **answer checkers** as our
previous reports (strict-match, flexible-extract); those checkers do not supply
the new generation prompt. GSM saved responses use **gsm8k_cot** strict/flexible
filters, with its `The answer is ...` strict pattern. Its installed YAML hash
must match the pinned prompt source. Additional GSM numeric/format fields are
project diagnostics. Reports include counts, means/medians of response tokens,
paired gains, exact McNemar p-values and paired bootstrap 95% CIs versus neutral.
Tests are unadjusted. Partial reports use explicitly counted matched pairs;
all completed deliveries remain in descriptive counts.

## Commands (new DIFFA machine)

```bash
git pull --ff-only
# Inspect the prompts/cohorts before generation, CPU only:
bash run_diffa_fewshot_emotions.sh --datasets bbh3 bbh5 --mode prepare
cat outputs/diffa-fewshot-emotions/bbh3/prompt.txt
cat outputs/diffa-fewshot-emotions/bbh5/prompt.txt

# BBH3 then BBH5, one model load. Waits for the shared GPU lock/idle GPU.
nohup bash run_diffa_fewshot_emotions.sh --datasets bbh3 bbh5 > outputs/diffa-fewshot-bbh.log 2>&1 </dev/null &

# Run later, separately:
nohup bash run_diffa_fewshot_emotions.sh --datasets gsm8k > outputs/diffa-fewshot-gsm.log 2>&1 </dev/null &

# CPU summaries (safe during generation):
bash run_diffa_fewshot_emotions.sh --datasets bbh3 bbh5 --mode summary
bash run_diffa_fewshot_emotions.sh --datasets gsm8k --mode summary
```

Rerunning the same command resumes only this condition's hash-bound saved answers.
Different output directories keep distinct experimental conditions separate.
