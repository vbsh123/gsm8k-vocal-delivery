# Kimi MultiArith: fresh 290-pair collection and 290-question evaluation

This is a separate experiment from DIFFA MultiArith and the selected BBH recovery
diagnostics. **L17 answer x1** is fixed before observing any Kimi MultiArith
answers. There is no layer/strength sweep, correctness filtering, base-checkpoint
comparison or reuse of DIFFA model responses/vectors.

The original MultiArith file has 600 rows but 580 whitespace-deduplicated questions.
The same pinned source, duplicate validation and seed42 split used previously are
reused: **290 collection questions / 290 disjoint test questions**. Every question
in each half is retained regardless of model success, classifier label, response
length or recording duration. Kimi processes full recordings through its native
audio pipeline. No question is shared between collection and test.

## Run on the Kimi machine

```bash
git pull --ff-only
nohup bash run_kimi_multiarith.sh \
  > outputs/kimi-multiarith290-L17.log 2>&1 < /dev/null &
tail -f outputs/kimi-multiarith290-L17.log
```

The CPU preparation freezes the split, prompt, L17 and x1, then reuses/resumes
`outputs/azure-multiarith580/` neutral/angry Azure recordings. On a machine without
these recordings, export `AZURE_SPEECH_KEY` and **`AZURE_SPEECH_REGION`** before
launching. Voice is Jenny, intensity 1.5; 580 questions × 2 styles = 1,160 clips.
Question wording and voice are matched. Existing compatible clips are not
regenerated, and the existing DIFFA result folder is not modified.

After audio preparation, the wrapper waits for current BBH Kimi follow-up,
full-steering and explicit-prompt queues, then the shared GPU lock. One cached
Kimi-Instruct model is used for collection and evaluation. No base/MOSS/Qwen
weights are needed or downloaded by this experiment.

### Collection

For all 290 collection pairs, generate neutral and angry answers with the same
pinned native Kimi greedy decoder, seed42 and 512-token limit. At L17, average
last-position answer-prediction outputs across each recording's forwards, then
subtract angry minus neutral. Average the 290 per-question differences equally.
Use that raw answer vector, with no normalization or fitted classifier.

L17 audio-token differences are also saved from the same forwards for reference,
but this specific test applies only the **answer** vector to answer positions.
The full 28-layer/encoded-feature traces are not stored in the pair files; only
L17 differences, raw collection responses, token counts, and audio hashes are
retained. Partial collection resumes per completed pair, and the frozen vector
file is preserved across resumes so old steered responses retain its same hash.

### Evaluation

For each of the other 290 questions, generate:

1. Unsteered neutral audio.
2. Unsteered angry audio.
3. The same neutral audio, with L17 answer vector ×1 on prefill and all cached
   answer-prediction steps.

There are **870 test generations** plus **580 collection generations**. All use
the same fixed math prompt:

```
Solve the math problem in the audio. Give concise step-by-step reasoning with only the necessary calculations. If the question asks for a total, compute that total. End with one plain-text line in exactly this format: #### <number>. Put a single numeric answer on that line, without Markdown, units, or separate component counts.
```

The primary comparison is numeric answer correctness of steered neutral versus
unsteered neutral on the full 290-question test set. Native installed upstream
GSM8K strict/flexible extraction and exact-match scoring are retained, with
additional numeric equivalence and format compliance. This is our spoken
adaptation using those components, not a dedicated official MultiArith task.
All raw answers are saved before scoring.

## Results and progress

```bash
.venv-diffa/bin/python run_kimi_multiarith.py --mode summary
```

The main accuracy/token-length table uses the same questions completed in all
three conditions. Paired comparisons show their exact matched N, wins/losses,
McNemar p-values and 95% question-paired bootstrap intervals. Secondary comparisons
remain exploratory/unadjusted. Token counts use Kimi's custom tokenizer with
`bos=False, eos=False`, excluding added start/end tokens.

Everything is saved under `outputs/kimi-multiarith290-L17/`: frozen selection and
run configuration, small per-pair `.pt` caches, both L17 vectors, raw test answers,
rescored answers, summary and scorer provenance. The original BBH and DIFFA
experiments stay separate.
