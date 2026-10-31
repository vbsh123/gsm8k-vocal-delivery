"""Prepare one style-neutral pair for extraction without copying audio."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("outputs/manual200-mixed/cropped"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vector-smoke/cropped"))
    parser.add_argument("--style", choices=["angry", "happy", "confident"], default="angry")
    args = parser.parse_args()
    source = args.path.resolve()
    output = args.output.resolve()
    if output == source:
        parser.error("Use a separate smoke output folder")

    recordings = {r["clip_id"]: r for line in (source / "manifest.jsonl").read_text().splitlines()
                  if line.strip() for r in [json.loads(line)]}
    results = {r["clip_id"]: r for line in (source / "results-neutral-binary.jsonl").read_text().splitlines()
               if line.strip() for r in [json.loads(line)]}
    result = next(r for r in results.values()
                  if r["style"] == args.style and r["predicted_style"] == "no")
    styled = recordings[result["clip_id"]]
    neutral = next(r for r in recordings.values()
                   if r["style"] == "neutral" and all(
                       r[key] == styled[key] for key in ("question_id", "speaker", "repeat")))

    output.mkdir(parents=True, exist_ok=True)
    for recording in (styled, neutral):
        target = output / recording["audio_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() and not target.is_symlink():
            target.symlink_to(source / recording["audio_path"])
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in (styled, neutral)))
    (output / "results-neutral-binary.jsonl").write_text(json.dumps(result) + "\n")
    print(f"Prepared {styled['clip_id']} and {neutral['clip_id']}")
    print(f"Smoke input: {output}")
    print(f"Vector output: {output.parent / 'style_vectors.pt'}")


if __name__ == "__main__":
    main()
