"""Remove spoken remarks using forced alignment; keep original recordings intact."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import unicodedata

from generate import append_jsonl, completed_records, render_preview, write_json


def alignment_words(text):
    text = unicodedata.normalize("NFKD", text.replace("’", "'")).encode("ascii", "ignore").decode().lower()
    words = re.findall(r"[a-z]+(?:'[a-z]+)*|\d+(?:[.,]\d+)*", text)
    # MMS's wildcard absorbs spoken numbers without guessing their pronunciation.
    return ["*" if word[0].isdigit() else word for word in words]


def question_parts(text, original):
    if not original or text.count(original) != 1:
        raise ValueError("Expected the complete original question exactly once in the spoken text.")
    prefix, _, suffix = text.partition(original)
    parts = [alignment_words(part) for part in (prefix, original, suffix)]
    if not parts[1]:
        raise ValueError("Question has no alignable words.")
    return parts


def crop_interval(parts, spans, seconds_per_frame, duration, padding):
    prefix, question, suffix = parts
    if len(spans) != sum(map(len, parts)) or any(not word for word in spans):
        raise ValueError("Alignment did not return a span for every word.")
    first = len(prefix)
    last = first + len(question) - 1
    question_start = spans[first][0].start * seconds_per_frame
    question_end = spans[last][-1].end * seconds_per_frame
    # Keep outer silence. At an added remark, pad only into the intervening gap.
    start = 0.0
    end = duration
    if prefix:
        previous_end = spans[first - 1][-1].end * seconds_per_frame
        start = max(question_start - padding, (previous_end + question_start) / 2)
    if suffix:
        next_start = spans[last + 1][0].start * seconds_per_frame
        end = min(question_end + padding, (question_end + next_start) / 2)
    if not 0 <= start < end <= duration:
        raise ValueError("Alignment produced invalid crop boundaries.")
    return start, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True, help="Generation folder containing manifest.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Separate folder for cropped WAVs and manifest")
    parser.add_argument("--limit", type=int, default=0, help="First N available recordings; 0 means all")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--padding-ms", type=float, default=40)
    args = parser.parse_args()
    if args.limit < 0 or args.padding_ms < 0:
        parser.error("limit and padding-ms must be nonnegative")
    source, output = args.path.resolve(), args.output.resolve()
    if source == output:
        parser.error("Use a separate output folder to preserve the original recordings")
    records = list(completed_records(source).values())
    if args.limit:
        records = records[:args.limit]
    config = {"source": str(source), "aligner": "torchaudio.pipelines.MMS_FA", "padding_ms": args.padding_ms}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "crop_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Crop settings differ; use a new output folder.")
    elif any(output.iterdir()):
        raise ValueError("Use an empty output folder for the first crop run.")
    write_json(config_path, config)

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model().to(args.device).eval()
    tokenizer, aligner = bundle.get_tokenizer(), bundle.get_aligner()
    completed = completed_records(output)
    failures = 0
    for index, record in enumerate(records, 1):
        clip_id = record["clip_id"]
        target = output / record["audio_path"]
        previous = completed.get(clip_id)
        if previous and target.is_file() and previous["source_audio_sha256"] == record["audio_sha256"]:
            print(f"[{index}/{len(records)}] skip {clip_id}", flush=True)
            continue
        print(f"[{index}/{len(records)}] crop {clip_id}", flush=True)
        try:
            original = record["original_question"]
            text = record["text"]
            parts = question_parts(text, original)
            audio, rate = sf.read(source / record["audio_path"], dtype="float32", always_2d=True)
            duration = len(audio) / rate
            start, end = 0.0, duration
            boundary_scores = None
            if text != original:
                waveform = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
                waveform = torchaudio.functional.resample(waveform, rate, bundle.sample_rate)
                with torch.inference_mode():
                    emission, _ = model(waveform.to(args.device))
                    spans = aligner(emission[0], tokenizer(sum(parts, [])))
                seconds_per_frame = waveform.shape[-1] / bundle.sample_rate / emission.shape[1]
                start, end = crop_interval(parts, spans, seconds_per_frame, duration, args.padding_ms / 1000)
                first, last = len(parts[0]), len(parts[0]) + len(parts[1]) - 1
                boundary_scores = [float(np.mean([s.score for s in spans[i]])) for i in (first, last)]
            start_sample, end_sample = max(0, int(start * rate)), min(len(audio), int(np.ceil(end * rate)))
            cropped = audio[start_sample:end_sample]
            if not len(cropped):
                raise ValueError("Crop would produce empty audio.")
            target.parent.mkdir(parents=True, exist_ok=True)
            if text == original:
                shutil.copy2(source / record["audio_path"], target)
            else:
                sf.write(target, cropped, rate, subtype="FLOAT")
            result = dict(record, text=original, uncropped_text=text,
                          source_audio_sha256=record["audio_sha256"], source_duration_seconds=duration,
                          crop_start_seconds=start_sample / rate, crop_end_seconds=end_sample / rate,
                          alignment_boundary_scores=boundary_scores,
                          sample_rate=int(rate), duration_seconds=len(cropped) / rate,
                          rms=float(np.sqrt(np.mean(cropped.astype(np.float64) ** 2))),
                          peak_amplitude=float(np.max(np.abs(cropped))),
                          audio_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            append_jsonl(output / "manifest.jsonl", result)
            completed[clip_id] = result
        except (ValueError, RuntimeError, OSError) as exc:
            failures += 1
            append_jsonl(output / "crop_errors.jsonl", {"clip_id": clip_id, "error": str(exc)})
            print(f"  FAILED: {exc}", flush=True)
    ready = [completed[r["clip_id"]] for r in records if r["clip_id"] in completed]
    render_preview(output, ready, completed)
    print(f"Saved {len(ready)} clips; {failures} failed. Listen: {output / 'listen.html'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
