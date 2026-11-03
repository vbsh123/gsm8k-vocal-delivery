"""Export the original Mistral GSM8K Speech test audio, without synthesis or cropping."""

import argparse
import io
import json
from pathlib import Path

from emotion_agreement import read_rows


DATASET = "mistralai/gsm8k_speech"


def parquet_rows(revision, files):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    for filename in files:
        print(f"Downloading/caching {filename}", flush=True)
        path = hf_hub_download(repo_id=DATASET, repo_type="dataset", filename=filename, revision=revision)
        with pq.ParquetFile(path) as parquet:
            source_row = 0
            for batch in parquet.iter_batches(batch_size=2, use_threads=False):
                for row in batch.to_pylist():
                    yield filename, source_row, row
                    source_row += 1


def export_rows(rows, output, revision, limit=None):
    import soundfile as sf

    completed = read_rows(output / "manifest.jsonl")
    (output / "audio").mkdir(parents=True, exist_ok=True)
    selected = 0
    skipped = []
    with (output / "manifest.jsonl").open("a") as stream:
        for source_index, (filename, source_row, row) in enumerate(rows):
            # Source audio row IDs are not assumed to match text GSM8K row IDs.
            clip_id = f"spoken_gsm_test_{source_index:05d}__original"
            if clip_id in completed and (output / completed[clip_id]["audio_path"]).exists():
                selected += 1
            else:
                payload = row["audio"]["bytes"]
                info = sf.info(io.BytesIO(payload))
                seconds = info.frames / info.samplerate
                if not 0 < seconds <= 30:
                    skipped.append({"source_index": source_index, "duration_seconds": seconds,
                                    "reason": "outside_shared_0_to_30_second_range"})
                    continue
                audio, rate = sf.read(io.BytesIO(payload), dtype="float32")
                relative = f"audio/{clip_id}.wav"
                # Preserve waveform samples and sampling rate; models resample on input.
                sf.write(output / relative, audio, rate, subtype="FLOAT")
                record = {"clip_id": clip_id, "question_id": f"spoken_gsm_test_{source_index:05d}",
                          "style": "original", "speaker": "source", "repeat": 0,
                          "audio_path": relative, "duration_seconds": seconds,
                          "sample_rate": rate, "answer": str(row["answer"]),
                          "source_dataset": DATASET, "source_revision": revision,
                          "source_index": source_index, "source_file": filename, "source_row": source_row}
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                completed[clip_id] = record
                selected += 1
                print(f"[{selected}] prepared {clip_id} ({seconds:.1f}s)", flush=True)
            if limit is not None and selected >= limit:
                break
    (output / "skipped.jsonl").write_text("".join(json.dumps(r) + "\n" for r in skipped))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/spoken-gsm-original"))
    parser.add_argument("--num-examples", type=int, help="First N eligible source rows; default all")
    args = parser.parse_args()
    if args.num_examples is not None and args.num_examples < 1:
        parser.error("--num-examples must be positive")
    from huggingface_hub import HfApi

    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "source.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if config["dataset"] != DATASET:
            parser.error("Output belongs to another dataset; use a fresh output folder")
    else:
        if (args.output / "manifest.jsonl").exists():
            parser.error("Existing manifest lacks source.json; use a fresh output folder")
        api = HfApi()
        revision = api.dataset_info(DATASET).sha
        files = sorted(name for name in api.list_repo_files(DATASET, repo_type="dataset", revision=revision)
                       if name.startswith("data/test-") and name.endswith(".parquet"))
        if not files:
            parser.error("Dataset has no data/test-*.parquet files")
        config = {"dataset": DATASET, "revision": revision, "files": files,
                  "max_seconds": 30, "audio": "original decoded samples; no crop or TTS",
                  "ids": "source audio row index, not text GSM8K ID"}
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    count = export_rows(parquet_rows(config["revision"], config["files"]),
                        args.output, config["revision"], args.num_examples)
    print(f"Prepared/reused {count} original clips <=30s: {args.output}")


if __name__ == "__main__":
    main()
