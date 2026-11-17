# Kimi BBH five-object deduction

Runs **both original and explicit-reasoning prompts**, sequentially, on identical
audio and the same split, with independent vectors and output directories.
For EACH prompt: full delivery evaluation is **250 neutral and 250 angry answers**. Own angry-minus-
neutral mean collected on a fixed seed42 first half (**125 questions**). Steering
at **L17 answer x1** is evaluated only on the other **125**. Collection questions
are included in the full delivery comparison, never in the steering comparison.
No question selection by correctness, classifier label, length or recording
duration. Full audio uses Kimi's native encoders, including long recordings.

```bash
git pull --ff-only
nohup bash run_kimi_bbh5.sh > outputs/kimi-bbh5-L17.log 2>&1 < /dev/null &
tail -f outputs/kimi-bbh5-L17.log
```

Azure variables, if missing clips need generation:

```bash
export AZURE_SPEECH_KEY='YOUR_KEY'
export AZURE_SPEECH_REGION='eastus'
```

The script reuses compatible Jenny intensity1.5 clips from
`outputs/azure-bbh-five250/` when available. Otherwise it generates/resumes **500
clips** in `outputs/azure-kimi-bbh-five250/`. Existing clips are linked; preserve
their source folders with any backup. Original question wording and original
five choices are retained, with option markers spoken as Option A–E. Source is
the pinned BIG-Bench-Hard revision already used in the DIFFA experiment; SHA256
is checked. Its exact split and prompt are frozen before Kimi results.

The original prompt is the established BBH-five spoken adaptation: task description,
audio question with five choices, "Let's think step by step", ending in
`The answer is (<letter>).` Collection and all evaluation conditions use that
same prompt, native greedy decoder, seed42 and 512-token limit. The vector is
an equal mean of 125 per-question angry-minus-neutral L17 answer-state means,
with raw magnitude retained. It is not a BBH-three or DIFFA vector. The NEW prompt
changes only "Let's think step by step." to "Give concise step-by-step reasoning
with only the necessary deductions." It collects its OWN mean from the same 125
pairs and tests on the identical other 125. Audio, final-answer requirement,
seed, decoder, budget, layer and strength stay the same across prompt variants.

CPU audio preparation may overlap current GPU work. Model work waits for Kimi
MultiArith, GSM8K and BBH queues, then the shared GPU lock. One cached Instruct
model is used; no base or other-model weights are downloaded.

The 250 unsteered collection generations supply the first-half delivery answers;
the other half requires 375 generations (neutral/angry/steered), for **625 per prompt,
1,250 total**. Both protocol selections are frozen before model inference.
Raw answers, per-pair caches, vector, audio hashes, token counts and configuration
are saved under `outputs/kimi-bbh5-collect125-test125-L17/` and
`outputs/kimi-bbh5-collect125-test125-L17-explicit-reasoning/`. Resume retains completed pairs
and answers. Original math results/configurations remain compatible.

```bash
.venv-diffa/bin/python run_kimi_bbh5.py --mode summary
.venv-diffa/bin/python run_kimi_bbh5.py --mode summary --explicit-reasoning
```

Two separate tables:

- All 250 matched neutral/angry questions: delivery effect.
- Other 125 matched neutral/angry/steered questions: held-out steering effect.

Both use the installed upstream `bbh_cot_zeroshot_logical_deduction_five_objects`
task's strict/flexible filters and exact-match metric. Format extractability is
an additional diagnostic that checks all five legal choices, independently of
the gold answer. Responses are scored verbatim. Reports include paired counts,
exact McNemar tests and 95% paired bootstrap intervals. Primary comparison is
held-out steered versus neutral flexible accuracy; secondary tests are
unadjusted. This is audio-adapted inference, not the official text-only prompting
protocol. L17 x1 was selected previously on BBH-three, not optimized on BBH-five.
