Rescore saved steering answers before extending CommonsenseQA. No scoring command loads a model or changes the raw results.

On the CommonsenseQA/evaluation machine, from the repository:

```bash
.venv-diffa/bin/python -u audit_commonsense_scoring.py
.venv-diffa/bin/python -u rescore_audio_steering.py --task commonsenseqa \
  --output outputs/commonsenseqa-independent-audit-v2
```

The first command compares strict/relaxed scoring for every available delivery and model. The second checks gold answers against the original checksum-verified CommonsenseQA source, and reports paired baseline/steering gains, McNemar p-values and 200,000-resample paired bootstrap confidence intervals. Full scoring disagreements are exported for inspection. The independent choice parser is not an official CommonsenseQA harness task.

The corrected custom relaxed scorer is `choice-relaxed-v2`. It accepts consistent repeated letters (`C: C`), `Final: C`, exact choice text followed by “is the correct answer”, and sentences such as “D. A pet store ...” without treating the article A as another option. Conflicting letters remain unparseable. Scoring accepts only explicit choices and never receives the gold answer during extraction. The main audit defaults to `outputs/commonsenseqa-scoring-audit-v2`, preserving the previous audit folder. Raw generation files and strict scoring are unchanged.

On the GSM8K/steering machine:

```bash
.venv-diffa/bin/python -u rescore_audio_steering.py --task gsm8k \
  --results outputs/steer-jenny-late-layers/original_mean/results.jsonl \
            outputs/steer-jenny-late-layers/third-cohort/original_mean/results.jsonl \
  --output outputs/gsm8k-L31-independent-audit
```

This downloads and freezes the current upstream lm-evaluation-harness revision and actual RegexFilter source, and uses its GSM8K flexible extraction rule. It reports string exact match and an additional numeric-equivalence sensitivity check (e.g. 12.00 equals 12), alongside the current scorer. It also checks saved labels against OpenAI's GSM8K test source. This is post-hoc scoring, not a full standard harness run with its text prompt/few-shot settings. Pass `--conditions` to select other saved conditions. Never pass overlapping copies of the same cohort.

To generate the remaining CommonsenseQA questions and evaluate the unchanged vector, leave the existing Azure environment variables set, then run on the evaluation machine:

```bash
nohup bash run_commonsense_remaining.sh \
  > outputs/commonsenseqa-remaining721.log 2>&1 < /dev/null &
tail -f outputs/commonsenseqa-remaining721.log
```

Sequence: rescore existing results; generate source indices 500–1220 in neutral only; wait for existing evaluation/GPU work; baseline plus existing L31 answer vector at strength 1, seed 42, fast_dllm/128 steps/32-token blocks/512 answer tokens/temperature 0; rescore new and combined held-out results. No vector extraction, no classifier/outcome filtering. Clips longer than DIFFA's 30-second input limit are listed in run_config.json and skipped, never cropped.

Defaults expect `outputs/commonsenseqa-angry-mean/{split.json,style-vectors.pt,results.jsonl}` and `outputs/azure-commonsenseqa500/{questions.jsonl,commonsenseqa-validation-source.jsonl}`. Preserve those folders. New audio lives in `outputs/azure-commonsenseqa-remaining721`, new answers in `outputs/commonsenseqa-remaining721-steering`.

There are 1,221 validation questions, of which the first 500 were already generated: 250 for extraction, 250 for held-out evaluation. The extension adds up to 721 questions / 1,442 generations. Combined held-out evaluation contains up to 971 distinct questions; the original extraction questions are excluded.

Mid-run accuracy, scoring sensitivity, p-values and intervals:

```bash
.venv-diffa/bin/python extend_commonsense_steering.py --summary-only
```

All confidence intervals and p-values are unadjusted. The original 250 and new 721 are also reported separately. Completed answers and successful Azure downloads are retained when restarting the same command. No model caches or recordings are deleted.
