"""Copy a random sample of externally unresolved clips into a portable listening folder."""

import argparse
import html
import json
from pathlib import Path
import random
import shutil


def read(path):
    with path.open() as stream:
        return {r["clip_id"]: r for line in stream if line.strip()
                for r in [json.loads(line)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("outputs/manual200-mixed/cropped"))
    parser.add_argument("--predictions", default="results-emotion2vec-evaluated.jsonl")
    parser.add_argument("--output", type=Path, default=Path("outputs/emotion2vec-unknown15"))
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    source = args.path.resolve()
    output = args.output.resolve()
    if source == output or output in source.parents:
        parser.error("Use a separate output folder, not the input folder or its parent")
    if output.exists() and any(output.iterdir()):
        parser.error("Output folder already contains files; use another --output")
    manifest = read(source / "manifest.jsonl")
    predictions = read(source / args.predictions)
    candidates = sorted(k for k, r in predictions.items()
                        if r["binary_prediction"] == "unresolved" and k in manifest)
    if not candidates:
        parser.error("No unresolved clips found in the prediction file")
    selected = random.Random(args.seed).sample(candidates, min(args.count, len(candidates)))
    (output / "audio").mkdir(parents=True, exist_ok=True)
    samples = []
    page = ['<!doctype html><meta charset="utf-8"><title>Audio review</title>',
            '<style>body{font:17px system-ui;max-width:900px;margin:30px auto;padding:15px}'
            'article{border-bottom:1px solid #ccc;padding:15px 0}audio{width:100%}</style>',
            '<h1>Audio review</h1><p>Check completeness, words and numbers, repetition, '
            'and the beginning/end of the question.</p>']
    for index, clip_id in enumerate(selected, 1):
        recording = manifest[clip_id]
        relative = f"audio/{index:02d}__{clip_id}{Path(recording['audio_path']).suffix}"
        shutil.copy2(source / recording["audio_path"], output / relative)
        samples.append({**recording, "audio_path": relative,
                        "source_audio_path": str(source / recording["audio_path"]),
                        "external_prediction": predictions[clip_id]})
        question = recording.get("original_question") or recording.get("text") or recording.get("question", "")
        page.append(f'<article><h2>Clip {index:02d}</h2>'
                    f'<audio controls preload="none" src="{html.escape(relative, quote=True)}"></audio>'
                    f'<p>{html.escape(question)}</p>'
                    f'<details><summary>Clip ID</summary>{html.escape(clip_id)}</details></article>')
    (output / "listen.html").write_text("\n".join(page), encoding="utf-8")
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in samples), encoding="utf-8")
    (output / "selection.json").write_text(json.dumps({
        "predictions": str(source / args.predictions), "seed": args.seed,
        "candidate_count": len(candidates), "clip_ids": selected,
    }, indent=2) + "\n")
    archive = shutil.make_archive(str(output), "zip", root_dir=output)
    print(f"Copied {len(samples)} of {len(candidates)} unresolved clips. Originals preserved.")
    print(f"Folder: {output}\nListen: {output / 'listen.html'}\nDownload: {archive}")


if __name__ == "__main__":
    main()
