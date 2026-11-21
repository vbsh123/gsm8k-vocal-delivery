# Phi L11/L30 double and L0/L11/L30 triple interventions

Two workers start immediately, without the shared GPU queue or an idle-GPU wait:

| Worker | Fixed interventions |
|---|---|
| 0 | L11 answer -1; L30 answer -2 |
| 1 | L0 audio -1; L11 answer -1; L30 answer -2 |

Each worker loads one persistent Phi model and evaluates its own combination
across MultiArith, GSM8K, SVAMP, BBH3, BBH5 and IFEval sequentially. They run
different combinations in parallel, not duplicate copies of one evaluation.
The existing concurrent-startup memory check and 18 GiB Torch cap apply per worker;
no unrelated processes are stopped.

```bash
git pull --ff-only
nohup bash run_phi4_l11_l30_parallel.sh \
  > outputs/phi4-L11-L30-parallel.log 2>&1 </dev/null &
tail -F outputs/phi4-L11-L30-parallel.log \
  outputs/phi4-L11-L30-double-triple/worker{0,1}.log
```

Original collection/test splits, audio, prompts, seed42 and native decoding are
preserved. IFEval uses 1024 tokens with official prompt/instruction scoring;
other tasks use 512 tokens and their existing scorers. Existing neutral baselines
are required and reused. Only new combined neutral-input responses are scheduled;
no new standalone intervention or unsteered emotional-delivery runs are made.

The source vectors use the original own-task angry-minus-neutral definition and
raw magnitude. Cached collections are shared, with collection locks preventing
duplicate capture writes; any missing collection captures are completed.
IFEval reuses its all-layer cache. L11 and L30 vectors are keyed by BOTH layer
and token group, so the two answer interventions never accidentally share one
direction. Source audio/config/vector receipts are stored with each job.

Outputs are under
`outputs/phi4-L11-L30-double-triple/<dataset>/{L11-L30,L0-L11-L30}`.
Old combinations remain saved. Resuming the same launcher skips saved answers
and collection work. The existing worker manager cleans up the sibling worker
if a child fails, preserving files already saved.

```bash
bash run_phi4_l11_l30_parallel.sh --mode summary
# Restrict the report, or the run, with --datasets:
bash run_phi4_l11_l30_parallel.sh --mode summary --datasets bbh3 bbh5 ifeval
```

Summaries run in the CPU scoring environment (`.venv-diffa`, override through
`DIFFA_PYTHON`) and include matched token means/medians, paired tests and CIs.
The standard `--only phi` backup includes this output root. This is an
exploratory fixed-combination follow-up chosen after prior layer/task results;
no success-dependent automatic selection or deletion is performed.
