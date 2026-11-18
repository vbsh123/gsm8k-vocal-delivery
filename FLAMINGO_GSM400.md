Audio Flamingo 3 GSM400 uses the existing pinned
nvidia/audio-flamingo-3-hf checkpoint and native Transformers model/processor.
The first400 canonical1269 GSM questions after seed42 shuffling match the earlier
Phi/MiniCPM collection/happy400 cohort. Evaluate800 neutral/angry answers with the
same original reasoning/#### prompt, greedy512 tokens, BF16/SDPA. No steering,
new TTS, transcription cascade or outcome filtering. AF-Think is not loaded.
The native processor windows full audio; over600-second inputs are rejected.
Audio-adapted inference, installed upstream GSM8K filters/metric on saved answers;
numeric equivalence and format extraction additional, all paired tests unadjusted.

```bash
nohup bash run_flamingo_gsm400.sh > outputs/flamingo-gsm400.log 2>&1 </dev/null &
tail -f outputs/flamingo-gsm400.log
```

This validates the existing GSM1269 source/audio before downloading anything.
The space preflight estimates30.8GiB for uncached checkpoint weights, up to8GiB
for a fresh isolated environment and4GiB reserve:42.8GiB if nothing is cached.
Installed environment bytes and completed snapshot shards reduce the estimate.
Insufficient disk exits before installation; no data is deleted. Environment
setup pins Python3.11/Torch2.9.1/cu128/Transformers5.0.0 in .venv-bbh-flamingo.
It waits for current Phi/Mini jobs and shared GPU lock, then runs alone. No
quantization/offload. CPU tests do not establish native GPU memory compatibility.

Outputs: outputs/gsm400-candidates/flamingo/original, including raw answers,
configs, scores, token lengths, paired summaries, runtime and scorer provenance.
Source audio/plans and older experiment outputs are retained.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python evaluate_gsm_candidates.py \
  --mode summary --model flamingo --prompt original --num-questions 400 \
  --output outputs/gsm400-candidates
python pack_experiment_backup.py --only flamingo --destination /workspace --samples 3 --sample-mb 100
```
