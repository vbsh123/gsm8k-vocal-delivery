# Object Counting: five conditions per question

Use all250 questions from the authors' pinned BIG-Bench Hard Object Counting
dataset. The exact source is checked against its Git blob ID before any TTS.
Every question is read unchanged by Azure Jenny in four styles: neutral, angry,
happy and sad. Jenny's happy maps to Azure `cheerful`, with styledegree1.5.
There are **1000 audio clips and 1250 DIFFA generations**, not five random samples
of the same condition.

| Condition | Audio | Prompt |
|---|---|---|
| neutral-reasoning | Neutral | Reasoning |
| neutral | The identical neutral recording | Bare |
| angry | Angry | Bare |
| happy | Happy | Bare |
| sad | Sad | Bare |

The bare prompt is:

```text
Answer the question in the audio. End your response with a final answer line in the format:
The answer is <number>.
```

The reasoning prompt is:

```text
Answer the question in the audio. Give concise step-by-step reasoning with only the necessary calculations. End your response with a final answer line in the format:
The answer is <number>.
```

The reasoning sentence comes from our earlier math prompt. These are custom audio
adaptations; this is not the complete historical math prompt or the harness text
generation template. The only difference between the two prompts is the selected
reasoning sentence. There are no worked examples, E2V/accuracy filters, reserved
subsets, activation fitting or steering.

The generation recipe stays the same in all five conditions: DIFFA fast_dllm,
seed42 reset per generation, 128 steps, block length32, temperature0, max512 new
tokens. Native chat wrapping and existing long-audio handling remain in place.
Full source/selection, prompts, audio hashes, model revision receipts and package
versions are saved. The runner resumes missing audio and condition-specific saved
answers, rejects incompatible caches, and waits for the shared GPU lock/idle GPU.

## Evaluator

Saved answers use the installed EleutherAI harness task
`bbh_cot_zeroshot_object_counting`, its strict/flexible extraction filters and
`exact_match` metric. Strict extraction accepts `The answer is 8.`; flexible
extraction selects the last numeric match. Although the upstream source contains
a number-word fallback, the pinned installed checker does not recover
`The answer is eight.` in our API test. We preserve that behavior and request
numeric answers. No response rewriting is performed. The task's CoT generation template is
not sent to DIFFA. A format preflight using the actual checkers runs before TTS.

Sources:

- [Dataset](https://github.com/suzgunmirac/BIG-Bench-Hard/blob/9ee07bd481feebf959a6b59d61ea57bdcf30964d/bbh/object_counting.json)
- [Harness task](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/bbh/cot_zeroshot/object_counting.yaml)

Report all completed condition accuracies and mean/median response tokens. Paired
comparisons use shared question IDs: neutral-reasoning vs neutral-bare, and each
bare emotional delivery vs neutral-bare. Include exact two-sided McNemar p-values
and paired bootstrap95% CIs, unadjusted. Partial comparisons show their actual N;
there are no emotion-to-emotion comparisons.

## Run

On a DIFFA machine with working Azure Jenny credentials exported:

```bash
git pull --ff-only
nohup bash run_object_counting_prompts.sh > outputs/object-counting-prompts.log 2>&1 </dev/null &
```

The command generates the1000 clips first, then evaluates the1250 conditions.
It needs no MAI access. Inspect progress or results:

```bash
tail -f outputs/object-counting-prompts.log
bash run_object_counting_prompts.sh --mode summary
```

CPU-only preparation (downloads public task metadata, writes TTS requests, tests
the answer checkers; no synthesis or GPU inference):

```bash
bash run_object_counting_prompts.sh --mode prepare
```

Audio/listening page: `outputs/azure-object-counting250/`.
Answers/summaries/prompts: `outputs/diffa-object-counting-prompts/`.

## Additional angry + reasoning condition

Use the exact existing `neutral-reasoning` prompt with all250 existing angry
recordings, in `outputs/diffa-object-counting-angry-reasoning/`. This adds250
generations without TTS, examples or steering, and does not alter the original
five-condition configuration or answers. It can be prepared while those answers
are still incomplete; comparisons use only matching completed question IDs.
The shared GPU lock queues this run after any active worker.

```bash
git pull --ff-only
nohup bash run_object_counting_angry_reasoning.sh > outputs/object-counting-angry-reasoning.log 2>&1 </dev/null &
bash run_object_counting_angry_reasoning.sh --mode summary
```

Reports compare angry-reasoning with neutral-reasoning (same prompt), angry-bare
(same audio), and neutral-bare. Each includes strict/flexible accuracy, mean/median
response tokens, matched N, paired gain, exact McNemar p and bootstrap95% CI.

## Inspect saved answers

Print three randomly sampled questions completed in all five original conditions,
with gold answers, full saved response text and token counts. Angry-reasoning is
included when prepared, with pending answers marked explicitly. CPU only; no
generation, scoring, outcome filtering or source-file changes:

```bash
python3 show_object_counting_outputs.py --count 3
python3 show_object_counting_outputs.py --longest --count 3
```

The second command deliberately selects the longest neutral-reasoning outputs to
inspect whether any intermediate steps appear; it is not a representative sample.
