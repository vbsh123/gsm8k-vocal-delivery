# MultiArith collection and held-out steering

Use `run_multiarith_mean.py` on the DIFFA steering machine. It collects a new
MultiArith angry-minus-neutral mean at L31 answer tokens, strength 1. It does
not reuse the GSM8K or SVAMP vectors.

The 600-row MultiArith source is pinned to the Zero-Shot-CoT authors' repository,
revision `5ef330fcdeec0cd26aee27943504f91f8ec1c33c`, with a checked SHA256.
Twenty rows repeat another question exactly after whitespace normalization.
Keep the first occurrence, retaining all original IDs in provenance, leaving
580 unique questions. A seed42 shuffle fixes **290 collection / 290 evaluation**
questions before outcomes. No emotion-classifier or correctness filtering is used.
`selection.json` freezes both halves, source provenance, prompt and intervention
before TTS. Changed settings require a new output folder.

Collection and evaluation use the same audio-adapted prompt requesting
`#### <number>`. The vector averages angry-minus-neutral answer activations over
matched forwards within each pair, then gives every collection question equal
weight. Only neutral and angry recordings from the collection half enter it.
The evaluation half runs neutral, angry, and steered neutral on the same questions,
with seed42 and unchanged 512-token / 128-step / 32-token-block decoding.
Long audio uses the existing full-audio chunked Whisper/global-adapter path.

Primary metric: **flexible numeric accuracy**. Also report strict `####` accuracy,
flexible string accuracy and format compliance. Scoring uses installed upstream
GSM8K task filters/metrics on MultiArith references, plus our numeric equivalence.
This is an audio adaptation, not an official MultiArith text-harness evaluation.
The summary reports paired confidence intervals and exact McNemar tests against
neutral; comparisons are unadjusted. This split is held out from vector collection;
pretraining overlap and overlap with earlier benchmark collections are not audited.

Inspect/freeze the source and split without Azure calls or GPU inference:

```bash
python run_multiarith_mean.py --mode prepare
```

With the existing environment and `AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION`
exported, launch from the repository root:

```bash
nohup .venv-diffa/bin/python -u run_multiarith_mean.py \
  > outputs/multiarith-mean.log 2>&1 < /dev/null &
tail -f outputs/multiarith-mean.log
```

Up to 1,160 Azure clips, 580 collection generations and 870 evaluation generations.
TTS can run alongside GPU work. Inference waits for reasoning-transfer,
mean-steering, strength-sweep and seven-object queues, then the shared GPU lock
and active GPU jobs. Existing queues do not wait on this runner.

Rerun the same command to resume audio, pair caches and completed evaluation
responses. A per-output lock rejects duplicate launches. Outputs are separate:
`outputs/azure-multiarith580` for audio and the cached source;
`outputs/multiarith580-angry-mean` for selection/configuration, collection pairs,
`style-vectors.pt`, raw responses, rescored responses and `summary.json`.

```bash
.venv-diffa/bin/python run_multiarith_mean.py --mode summary
```
