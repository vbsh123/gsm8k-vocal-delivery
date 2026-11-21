# GSM Final answer comparison

`diffa_gsm_final_answer.py` is a separate GSM-only run, using the exact saved
happy collection prompt for all 718 held-out neutral, happy, and angry deliveries
and happy L31 answer steering at +1. It preserves the previous #### run.
It keeps the existing question split, full audio, seed 42, 512-token budget,
128 steps, block length 32, and fast-dllm decoding.

The existing happy vector was already collected using this prompt, so its
425-pair collection is reused. Delivery answers are newly generated in the new
output directory; previous #### responses are never imported. Steering reuses
the newly generated neutral baseline. No other datasets, transfers or strengths
are scheduled. The script waits on the shared DIFFA GPU lock and idle GPU.

Use the same original `scoring.score_response` scalar-final-v2 scorer across all
conditions. These are original numeric correctness results, not upstream ####
strict-match scores. Reports use the shared completed questions and include
paired unadjusted tests and mean/median token counts. The new prompt comparison
is exploratory because the previous experiments have already been inspected.

```bash
nohup bash run_diffa_gsm_final_answer.sh > outputs/diffa-gsm-final-answer.log 2>&1 </dev/null &
bash run_diffa_gsm_final_answer.sh --mode summary
bash run_diffa_gsm_final_answer.sh --mode collection-summary
```

Collection summary scores the actual saved neutral/happy collection responses
without inference. It adds archived angry responses where the same IDs, Jenny
voice, prompt, budget, seed, decoder and references are available. Archived angry
provenance is identified separately; absent angry answers are shown as missing.
These 425 fitting questions remain separate from the 718 held-out questions.
