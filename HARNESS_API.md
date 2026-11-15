Installed harness task API and saved-answer audit
================================================

We now use the installed EleutherAI lm-evaluation-harness package, pinned to
`d6de81643928d653435c431bae19945d41d32520`. Active scoring no longer extracts
Python AST nodes, removes decorators/base classes, or executes copied classes.

```bash
git pull --ff-only
bash setup_harness.sh
.venv-diffa/bin/python -u audit_harness_api.py \
  > outputs/harness-api-audit.log 2>&1
tail -n 60 outputs/harness-api-audit.log
```

This is CPU-only rescoring. It can run while the five-object BBH generation
continues; no generation needs restarting. Original response and summary files
are retained. The setup installs the base harness without an HF/vLLM model
backend, and constrains dependencies using `requirements-diffa.txt`. It does not
install another PyTorch or load a model. The pinned dependency combination was
tested in a separate Python 3.10 environment; the actual GPU machine runs setup
and a task-API smoke check before the audit.

Default audit inputs:

- BBH-three delivery run: `outputs/bbh-logical200-diffa2-harness-chunked`.
- BBH-three 125/125 steering: `outputs/bbh-logical250-angry-mean`.
- SVAMP aligned-prompt delivery run: `outputs/svamp500-diffa2-hashes`.

Only runs present on the machine are rescored. Override paths with `--help`.
Output directories under `outputs/harness-api-audit/` contain:

- `summary.json`: new accuracy, paired gains, McNemar p-values and confidence intervals.
- `rescored.jsonl`: raw responses plus new scores and extracted predictions.
- `previous-summary.json`: a snapshot of the prior summary, when present.
- `comparison.json`: previous/current counts for matching metric/condition rows.
- `svamp-hashes/disagreements.jsonl`: SVAMP per-answer score disagreements.

For BBH-three steering, the earlier raw answers did not store per-answer harness
scores, so comparison is against the saved aggregate summary. The new rescored
file contains per-answer scores for inspection. Different cohort sizes between a
stale summary and current responses are reported, not treated as proof of a
scorer change.

How the Python API is used
--------------------------

`HarnessTaskScorer` in `rescore_audio_steering.py` reads the installed upstream
task YAML with the harness's YAML loader. It instantiates `ConfigurableTask`,
supplies the already-selected documents using its `custom_dataset` extension,
and calls `build_all_requests()`. The saved response is assigned to each request;
`apply_filters()` and `process_results()` perform upstream extraction and
exact-match scoring. No model is called. Task source hashes, installed version,
and the installation's Git commit are recorded in `api-provenance.json`.

BBH has dedicated upstream tasks for logical deduction with three/five/seven
objects. `HarnessScorer` in `score_bbh_harness.py` selects the corresponding task.
GSM8K uses the upstream `gsm8k` task. This pinned harness has no dedicated SVAMP
task: SVAMP is explicitly scored with GSM8K numeric filters/metric against
SVAMP gold answers. `flexible-numeric` is a separate custom numeric-equivalence
metric and must not be described as an official SVAMP metric.

This is official task scoring on our spoken adaptation, not the full official
text inference protocol: our audio input, generation prompt, zero-shot context,
and DIFFA decoder remain as recorded. We do not change prompts, rerun inference,
or retroactively apply generation stop strings during the audit.

Our older `harness_filter` helper now imports the installed `RegexFilter`
normally. Legacy standalone SVAMP and saved-GSM8K audits retain their explicitly
documented normalized-string/numeric comparisons; the new audit and current
BBH/GSM8K transfer scorers use `ConfigurableTask.process_results()` directly.

Verification
------------

The old source loader and installed task API were compared on 2,406 locally
available saved GSM8K neutral/angry responses. Strict/flexible extracted
predictions and official exact-match scores had zero disagreements. This is a
scorer comparison, not a new accuracy experiment. The complete BBH/SVAMP runs
must be audited on the machine holding their raw responses using the command
above. The test suite also covers original-file preservation and the standard
task API's use without dataset download or model inference.
