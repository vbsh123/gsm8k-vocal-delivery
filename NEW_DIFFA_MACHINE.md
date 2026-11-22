# New A40 machine

Clone into the exact same path as the original machine. Frozen configs contain
absolute paths and hashes, so do not edit or relocate those configs.

On the original DIFFA machine:

```bash
git pull --ff-only
.venv-diffa/bin/python pack_diffa_migration.py
```

Download the printed `diffa-agreed-runs-*.zip` and its `.zip.sha256`. Upload them
alongside the existing `jenny-six-datasets-audio-*.zip` and its checksum to
`/workspace` on the new machine. Audio is not included in the migration bundle.

The migration bundle includes the new shared GSM angry/happy run, earlier
angry-agreed BBH and remaining-task runs, frozen configurations, saved answers,
completed vectors, running aggregate checkpoints, and source dependencies.
It preserves each protocol in its original directory. An in-progress aggregate
is a point-in-time resumable checkpoint, not a completed vector. Continuing a
snapshot on another machine does not stop the original process: assign one
machine as the owner of each output directory and keep results separate.

On the new machine (Python 3.10–3.12, compatible NVIDIA driver):

```bash
cd /workspace
git clone https://github.com/vbsh123/gsm8k-vocal-delivery.git
cd /workspace/gsm8k-vocal-delivery
bash setup_new_diffa_machine.sh
```

The script detects a single audio ZIP and optionally a single migration ZIP in
`/workspace`. If there are several, pass the exact paths as arguments. Uploads
must be complete. Checksums are verified when their sidecars are present; ZIP
CRC checks also run while extracting. Existing different files are never
overwritten. The existing repository-relative audio paths are preserved.

Setup installs the pinned DIFFA revision and PyTorch/CUDA package versions,
verifies the pinned harness, and compares important packages with the original
machine receipt. Model weights download on first model load. No experiment or
TTS is launched. A separate emotion-classifier environment is unnecessary for
using already classified recordings and collecting vectors. Leave enough disk
space for both uploaded ZIPs, extracted audio, environments, and model weights;
100 GiB is a practical starting allocation for this scoped setup, not the older
machine's unused TTS/model caches.

If a vector bundle arrives after initial setup:

```bash
python3 restore_diffa_bundle.py /workspace/diffa-agreed-runs-*.zip
.venv-diffa/bin/python check_diffa_migration.py
```

Inspect version differences before comparing across machines. Then run the
small neutral replay, which includes the longest available clip:

```bash
bash -c 'export HF_HOME=/workspace/.hf_home; export HF_HUB_CACHE=$HF_HOME/hub; export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub; .venv-diffa/bin/python -u replay_diffa_baseline.py --run'
```

The replay compares ten raw responses against the snapshot's saved neutral
responses under the unchanged prompt, audio hashes, seed, budget and decoder.
It uses the shared GPU queue and performs no steering. An identical small replay
is a useful check, not a guarantee of full cross-machine reproducibility. A
mismatch stops with details in `outputs/new-machine-neutral-replay/results.json`.
Do not start full comparisons until any software or replay mismatch is understood.
