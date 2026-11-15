Two additional BBH reasoning tasks, no steering
==============================================

On the IFBench machine, export Azure Speech credentials in the shell, then:

```bash
git pull --ff-only
nohup bash run_bbh_after_ifbench.sh \
  > outputs/bbh-after-ifbench.log 2>&1 < /dev/null &
tail -f outputs/bbh-after-ifbench.log
```

The launcher waits for IFEval and IFBench workers/launchers to exit, sets up the
pinned local harness library, and then runs two tasks sequentially:

| Task | Questions | Deliveries | Generations |
|---|---:|---|---:|
| Web of Lies | 200 | neutral, angry, happy | 600 |
| Tracking Shuffled Objects, three objects | 200 | neutral, angry, happy | 600 |

Total: 1,200 Azure recordings and 1,200 DIFFA-2 generations. No vector is collected
or loaded; there is no steering. Each task's model process exits before the next
starts. Both use the shared GPU lock and wait for active GPU work to exit.

Selection uses a fixed seed42 shuffle of the official 250-question task. Source
revision and per-task SHA256 checksums are pinned. Original question wording is
unchanged except tracking option markers `(A)` become spoken `Option A`, as in
our previous BBH experiments. Gold answers are metadata, never spoken. All three
styles use identical questions, Jenny voice, and Azure intensity 1.5; happy maps
to Azure cheerful. No outcome or classifier filtering.

The prompts request `The answer is yes.` / `The answer is no.` for Web of Lies,
and `The answer is (A).` with the selected letter for tracking. Both formats are
checked against the actual installed task API before any TTS spending.
Scoring uses dedicated EleutherAI BBH tasks, including their strict/flexible
filters and exact-match metrics, not GSM8K or custom choice parsing. Flexible
extraction is primary, angry versus neutral is the primary contrast, and happy
is a secondary delivery. All results are reported, including null results.

DIFFA uses seed42 reset for every generation, temperature0, 128 steps, block32,
512 answer tokens and fast_dllm. Audio is never cropped or stretched: <=30-second
clips use native encoding, longer clips use the same experimental chunked
Whisper/global-adapter encoding as our existing BBH run. The mode is recorded
for every answer. This is a spoken adaptation, not the unmodified text/few-shot
inference protocol.

Data: `outputs/azure-bbh-reasoning200/<task>/`.
Results: `outputs/bbh-reasoning200-diffa2/<task>/`.
Each result directory contains frozen `run_config.json`, `results.jsonl`,
`rescored.jsonl`, `summary.json`, and API provenance. Existing audio and answers
resume individually after interruption. Changing the selected questions or
prompt requires another output/source folder.

Check both tasks while they run, without loading DIFFA:

```bash
.venv-diffa/bin/python run_bbh_reasoning200.py --mode summary
cat outputs/bbh-reasoning200-diffa2/status.json
```

The summary uses questions completed in all three deliveries and reports paired
McNemar p-values and 95% paired bootstrap confidence intervals, unadjusted.
To rerun just inference from prepared audio/configs:

```bash
.venv-diffa/bin/python -u run_bbh_reasoning200.py --mode evaluate
```

SVAMP tolerance check (separate machine)
---------------------------------------

No new SVAMP prompt or model generation is needed. The existing `####` responses
can be rescored with the authors' numerical criterion: absolute answer error
<=0.1. This is an additional metric, not the authors' equation-parsing pipeline.
Exact numeric correctness remains primary. Rerun the saved-answer audit:

```bash
git pull --ff-only
.venv-diffa/bin/python -u audit_harness_api.py \
  > outputs/harness-api-audit.log 2>&1
tail -n 45 outputs/harness-api-audit.log
```

Look for `author-tolerance`. The current SVAMP steering queue's summary also
includes it after pulling this update, including scores for already saved
answers. Original response files and generation prompts remain unchanged.
