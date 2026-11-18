Happy versus neutral on Phi-4 and MiniCPM uses the first400 canonical GSM1269
questions shuffled seed42, exactly the400 angry-neutral collection questions.
This is an exploratory delivery comparison, not held-out steering validation.
No steering, vector collection, outcome selection or classifier filtering.

Both deliveries receive the same original reasoning/#### prompt and native
greedy512-token generation settings, as used by each model's earlier collection.
Full audio is supplied to each model's native processor; no cropping or custom
feature joining. Scoring uses installed upstream GSM8K filters/metric, with
numeric equivalence and format extraction as additional diagnostics. It is not
full official text-harness inference. Paired p-values/intervals are unadjusted.

`run_gsm_happy400.sh` prepares only missing neutral/happy Azure audio, first
reusing compatible request/audio files from existing plans. Missing TTS requires
AZURE_SPEECH_KEY and AZURE_SPEECH_REGION (singular). Source request plans and
existing results are not modified. The new audio source is outputs/azure-gsm-happy400.

The wrapper waits for current Phi/Mini queues, then runs MiniCPM and Phi serially
under the shared GPU lock and active-GPU check. Existing neutral collection
responses are copied only after checking model, revision, prompt, decoder,
question, voice, reference and audio identity; source hashes are recorded. Missing
neutral responses are generated normally. Resuming preserves complete answers.

```bash
nohup bash run_gsm_happy400.sh > outputs/gsm-happy400.log 2>&1 </dev/null &
tail -f outputs/gsm-happy400.log
```

CPU progress/scoring, without a GPU lock:

```bash
for model in minicpm phi4; do
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv-diffa/bin/python evaluate_gsm_happy400.py --mode summary --model "$model"
done
```

Outputs live in outputs/gsm-happy400/{minicpm,phi4}: frozen split/config, all800
raw answers/model, response tokens, baseline reuse provenance, runtime, rescored
answers and paired summaries. Focused Phi/Mini experiment backups include these
outputs and samples from the neutral/happy audio source.
