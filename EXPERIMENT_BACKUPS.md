# Downloadable experiment evidence

Run on each GPU machine now, without stopping ongoing experiments:

```bash
cd /workspace/gsm8k-vocal-delivery
git pull --ff-only
.venv-diffa/bin/python pack_experiment_backup.py
```

The command writes `experiment-evidence-<hostname>-<UTC timestamp>.zip` under
`/workspace`, outside `outputs`, and prints its full path and SHA256. Download
that ZIP and its `.zip.sha256` sidecar from each machine. Names identify the
machine and snapshot time. The command only reads source files: it never deletes,
truncates or restores anything, invokes no model and makes no network requests.
It needs only the Python standard library and the Git CLI.

Included:

- JSON/JSONL raw answers, configs, plans, question banks, splits and summaries
  throughout surviving outputs and `recovery-backups`.
- Scorer provenance, cached text/YAML/Python scoring assets, logs and reports.
- Collected/fitted vectors, including `style-vectors.pt`. Model caches/weights
  and large intermediate pair-activation caches are excluded.
- The actual tracked source working tree, Git commit/diff, Python/package
  inventory and any downloaded IFEval checker source.
- Up to three question groups from each source plan and each config's collection
  and test section, keeping available styles together. Samples are capped at
  256 MiB before compression, with core result configurations prioritized.
  Symlink targets are read as real audio; the samples need no external symlinks.
- `MANIFEST.json`: file hashes, source paths, audio-sample index, artifact issues,
  core-run saved/planned answer counts and required-vector availability.

The archive is a point-in-time snapshot of a running experiment. Complete JSONL
records are retained; a partially appended last record is omitted only in the
archive. Changed/unreadable files and missing sample references are flagged.
Printed missing-run notices mean absent **on this machine**, not proof of loss
on all machines. Counts do not by themselves establish protocol validity.

ZIP CRC checks run before completion; the final ZIP also gets a SHA256 sidecar.
Keep the downloaded archive off-machine. A ZIP on the same GPU instance is not
an independent backup. Private keys, environment-variable dumps, shell history,
model downloads and virtual-environment directories are not included.

## Preserve complete original audio too

The small evidence ZIP supports rescoring and inspecting examples. Samples
alone cannot reproduce inference on every question. Azure regeneration is not
guaranteed byte-identical. For exact-input reruns, also preserve full original
audio on a durable store.

With enough disk space, run:

```bash
.venv-diffa/bin/python pack_experiment_backup.py --full-audio
```

This writes the evidence ZIP plus a separate `*-full-tts-audio.zip` and checksum.
The second archive includes all existing Azure/ElevenLabs audio referenced by
TTS plans, at every logical path (including reused symlink aliases). It excludes
downloadable model weights and derived speed variants without TTS plans. The
full audio archive can be several GiB; download the smaller evidence ZIP first.

Use `--destination /another/path/outside/outputs` to choose another archive
location; `--samples 5` increases sampled question groups and `--sample-mb 512`
raises the sample budget. Source files are unchanged by every option.

## Smaller backup of the newest results

To avoid repacking the earlier experiments already downloaded, focus on the
new MultiArith/IFEval reruns and MOSS/Kimi BBH evaluations:

```bash
.venv-diffa/bin/python pack_experiment_backup.py --only multiarith ifeval moss kimi --samples 5
```

Run on both machines. It includes whichever selected runs exist locally, their
source question/TTS plans, all saved raw responses, configs, summaries, vectors,
scorer metadata and matched audio samples. The source-code snapshot and root
logs remain included. It scans matching `recovery-backups` folders too. Missing
selected runs are reported but do not stop packaging. File names include the
selected groups, machine ID and timestamp. Pair caches and model weights stay
excluded. Add `--full-audio` to preserve all TTS-plan recordings for these selected
groups in a second ZIP.
