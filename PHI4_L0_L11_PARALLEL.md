# Phi L0 audio -1 + L11 answer -1

Two queued workers with disjoint task assignments:

| Worker | Sequential tasks | Test questions |
|---|---|---|
| 0 | MultiArith, GSM8K | 290, 869 |
| 1 | IFEval, BBH3, BBH5 | 254, 125, 125 |

The workers run in parallel, each loading one Phi model and reusing it for its
whole sequence. The parent waits on the shared GPU queue and idle GPU before
starting both processes. Each process uses the existing 18 GiB Torch cap.

```bash
git pull --ff-only
nohup bash run_phi4_l0_l11_parallel.sh \
  > outputs/phi4-L0-L11-parallel.log 2>&1 </dev/null &
tail -F outputs/phi4-L0-L11-parallel.log \
  outputs/phi4-L0audio-minus1-L11answer-minus1/worker{0,1}.log
```

Each task preserves its existing collection/test split, prompts, audio hashes,
seed42 and native generation recipe. IFEval retains 1024 tokens and official
prompt/instruction scoring; all other tasks retain512 tokens. Only combined
neutral-input responses are generated. Existing neutral baselines are required
and reused; no standalone layer evaluations are scheduled.

Compatible L0/L11 vectors are reused through the existing collection caches.
IFEval uses its all-layer collection cache. Missing collection captures are
completed on the full original collection half; a smaller subset mean cannot
replace the original full-half vector. Directions remain raw angry-minus-neutral
means, multiplied by -1 at both interventions. The L0 audio hook acts on prefill
audio positions; L11 answer steering acts on native answer prediction forwards.

Configs, vector provenance, raw responses, worker logs and summaries are under
`outputs/phi4-L0audio-minus1-L11answer-minus1/<task>/combined`. Earlier SVAMP,
L0/L30 and standalone results remain saved. Resuming the same command skips saved
captures and answers. Worker failures stop the sibling process while preserving
saved work. This follow-up was selected after examining prior layer/task results
and remains exploratory, with unadjusted paired tests.

```bash
bash run_phi4_l0_l11_parallel.sh --mode summary
```

Summary uses the CPU scoring environment (`.venv-diffa`, override with
`DIFFA_PYTHON`), not another GPU model. It reports each task separately and includes
matched response mean/median token counts. The standard `--only phi` evidence
backup includes these new outputs.
