# Phi GSM8K combined steering

Use the existing400 collection/869 test split, original prompt, seed42 and greedy512-token budget. Apply the raw angry-minus-neutral direction at **L0 audio ×−1** during prefill and **L30 answer ×−2** at the last prediction position on every forward, in the same native generation. This is an exploratory test after inspecting layers and strengths.

The runner copies the exact vectors from `outputs/phi4-sweeps/gsm8k/L0-audio-a-1` and `outputs/phi4-gsm-reverse400-869-L30-a-2`. Compatible existing neutral and individual steering answers are reused with provenance; no new TTS or vector collection. Four conditions share the same869 neutral recordings. Ordinarily only869 combined answers remain. Interrupted runs resume; another process cannot write this output simultaneously.

```bash
git pull --ff-only
nohup bash run_phi4_gsm_combo.sh > outputs/phi4-gsm-combo.log 2>&1 </dev/null &
tail -F outputs/phi4-gsm-combo.log
```

No shared GPU lock or queue wait; checks available memory and caps Torch at18GiB. Can run alongside the existing IFEval worker when memory permits.

```bash
bash run_phi4_gsm_combo.sh --mode summary
```

Summary uses the intersection completed in all four conditions. It reports upstream GSM8K strict/flexible accuracy, numeric equivalence, format compliance, response-token mean/median, and paired comparisons of the combination against neutral and each individual intervention. Comparisons are exploratory and unadjusted. Results/vectors/receipts are included in the Phi backup group.

If the combination is unhelpful, separately test **+1 answer steering alone** across all32 layers (no L0 intervention in this fallback):

```bash
nohup bash run_phi4_sweep.sh --datasets gsm8k --layers $(seq 0 31) \
  --group answer --strengths 1 --no-wait \
  > outputs/phi4-gsm-answer-plus1.log 2>&1 </dev/null &
bash run_phi4_sweep.sh --mode summary --datasets gsm8k --layers $(seq 0 31) \
  --group answer --strengths 1
```

The fallback runs layers sequentially in one worker and reuses the400-question all-layer vector collection and neutral answers. It is intentionally not started automatically: inspect the combined result first.

## MultiArith alongside GSM8K

The same runner accepts `--dataset multiarith`. It uses MultiArith's own290 collection/290 test split and own vectors: the existing L0 audio−1 sweep and original L30 answer-vector run. It reuses compatible neutral/audio-only controls and generates missing L30 answer−2 controls as well as combined answers (normally580 new generations). It never substitutes GSM vectors. Thus the combination can be compared against both standalone interventions on identical questions.

```bash
nohup bash run_phi4_gsm_combo.sh --dataset multiarith \
  > outputs/phi4-multiarith-combo.log 2>&1 </dev/null &
bash run_phi4_gsm_combo.sh --dataset multiarith --mode summary
```

GSM8K and MultiArith have separate output folders and process locks, with no shared GPU wait. Start each once; the original GSM command remains valid.

## SVAMP

```bash
nohup bash run_phi4_svamp_combo.sh > outputs/phi4-svamp-combo.log 2>&1 </dev/null &
bash run_phi4_svamp_combo.sh --mode summary
```

Uses the original500 collection/500 test split. Reuses the collected L0 audio and L30 answer vectors and existing neutral answers, then generates only500 combined responses. No standalone control generations, TTS, or collection stages are launched. The L30 vector comes from the completed answer−1 run; the same raw vector is scaled by−2 for this combination. Output: `outputs/phi4-svamp-L0audio-minus1-L30answer-minus2-combined-only`. Summary compares combined against reused neutral answers. A missing neutral baseline causes an error before model load rather than new baseline generation. GSM and MultiArith four-condition runs are unchanged.

SVAMP L0 audio−1 + L30 answer−1 uses the same vectors/baseline and a separate output folder:

```bash
nohup bash run_phi4_svamp_combo.sh --answer-strength -1 > outputs/phi4-svamp-combo-minus1.log 2>&1 </dev/null &
bash run_phi4_svamp_combo.sh --answer-strength -1 --mode summary
```
