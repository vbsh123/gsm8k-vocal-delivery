This experiment collects angry-minus-neutral directions on the same400 canonical
Phi GSM collection pairs, then uses the remaining869 questions as the test pool.
Collection and test IDs are disjoint. No TTS or audio alterations.

1. Validate old collection config and400 pair caches; freeze the new config and
   retain raw old responses plus per-file hash receipts.
2. Wait ONLY for outputs/phi4-gsm-extra-L17-a-2.lock. No global GPU lock or
   Flamingo queue. Phi uses20GiB Torch cap and requires22GiB free immediately.
3. Replay800 neutral/angry native generations with all32 decoder hooks in each
   forward. Collect both answer-position means across every generation forward
   and audio-position means on initial prefill. No teacher forcing or gold input.
   Raw per-question angry-minus-neutral differences are averaged equally over400.
   Generation responses must match original collection responses; discrepancies
   stop the run with caches retained for inspection.
4. Reuse compatible neutral test responses from the completed869-question Phi
   run and available angry responses from the earlier original-prompt full GSM
   evaluation. Generate missing neutral/angry references on the869 test pool.
   Source reuse checks prompt/model/decoding, question/voice, audio/reference
   identity and zero interventions. No regenerated training outcome filters.
5. CPU installed upstream GSM8K scoring selects test cases with neutral numeric
   incorrect and angry numeric correct. Save challenge IDs and reference hashes.
6. Test each of32 layers separately, at strength-1, for answer and audio groups
   separately. Answer injection modifies only the last prediction state each
   forward. Audio injection modifies native audio positions on prefill only;
   subsequent decoding uses native KV caching. Audio group means positions in
   the language decoder, not a separate encoder-layer sweep.

The model uses the original reasoning/#### prompt, pinned Phi revision, BF16
SDPA, greedy512-token decoding, seed42 and original GenerationConfig throughout.
Hooks preserve cache/tuple output and are removed in finally blocks. Collection
and steering use identical group definitions. All-layer collection requires new
forwards because earlier400-pair caches contain only L17/L30, not all layers.

The recovery cases are OUTCOME-SELECTED: neutral numeric0% and angry100% are by
construction. This is layer selection/diagnosis, not unbiased benchmark accuracy.
Reports show per-layer/group percentages and completion counts, without p-values
for the selected cohort. An independently frozen evaluation would be a separate
experiment. Full test reference scoring and all raw generations are retained.

```bash
nohup bash run_phi4_all_layer_diagnostic.sh > outputs/phi4-all-layer-diagnostic.log 2>&1 </dev/null &
tail -f outputs/phi4-all-layer-diagnostic.log
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python phi4_all_layer_diagnostic.py --mode summary
```

Output: outputs/phi4-gsm400-all-layers-test869-minus1. Resumable atomic pair
caches store raw responses and all-layer means (~600MiB for400 pairs). Vectors,
references, frozen challenge, diagnostic results, scores, provenance and runtime
metadata are included in focused Phi evidence backups. Concurrent memory caps
exclude non-Torch allocations; native GPU execution remains to be verified.

Flamingo can be launched without its queue using run_flamingo_gsm400.sh
--concurrent. That mode requires32GiB free and caps Torch at30GiB. It does not
stop Phi or use the shared GPU lock. stop_waiting_flamingo.py stops only the
old wrapper when all its children are flock waiters, and refuses if evaluation
is already running. Pinned native checkpoint/model/decoding are unchanged.
