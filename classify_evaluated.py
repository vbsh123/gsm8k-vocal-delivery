"""Save external emotion labels and math accuracy groups for already-evaluated clips."""

import argparse
from collections import defaultdict
import importlib
import json
import math
from pathlib import Path

from analyze_steering_groups import flexible_correct, read_rows


def accuracy_rows(predictions, answers):
    groups = defaultdict(list)
    for clip_id, prediction in predictions.items():
        if clip_id not in answers:
            continue
        answer = answers[clip_id]
        for style in ("all", prediction["style"]):
            groups[style, "binary", prediction["binary_prediction"]].append(answer)
            groups[style, "emotion", prediction["emotion"]].append(answer)
    output = []
    for (style, kind, label), rows in sorted(groups.items()):
        n = len(rows)
        row = {"generated_style": style, "group_type": kind, "label": label, "count": n}
        for scoring, score in (("strict", lambda r: r["correct"]), ("non_strict", flexible_correct)):
            k = sum(score(r) for r in rows)
            a = k / n
            margin = 1.96 * math.sqrt(a * (1-a) / n)
            row[scoring] = {"correct": k, "accuracy": a,
                            "ci_low": max(0, a-margin), "ci_high": min(1, a+margin)}
        output.append(row)
    return output


def print_accuracy(rows, by_style=False):
    print("\nMath accuracy grouped by external predictions (clips, not unique questions)")
    print(f"{'Generated style':16} {'Grouping':9} {'Label':16} {'Clips':>6} {'Strict':>9} {'Non-strict':>11}")
    for r in rows:
        if r["generated_style"] != "all" and not by_style:
            continue
        print(f'{r["generated_style"]:16} {r["group_type"]:9} {r["label"]:16} '
              f'{r["count"]:6} {r["strict"]["accuracy"]:9.1%} {r["non_strict"]["accuracy"]:11.1%}')
    print("Binary and individual-emotion tables describe the same clips; do not add their counts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("emotion2vec", "sensevoice", "speechbrain"))
    parser.add_argument("--path", type=Path, default=Path("outputs/manual200-mixed/cropped"))
    parser.add_argument("--results-file", default="results-reasoning-v2.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--by-style", action="store_true", help="Also print groups within each generated style")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    answers = {r["clip_id"]: r for r in read_rows(args.path / args.results_file)}
    manifest = {r["clip_id"]: r for r in read_rows(args.path / "manifest.jsonl")}
    recordings = {k: r for k, r in manifest.items() if k in answers}
    print(f"Evaluated clips absent from manifest: {len(answers.keys()-manifest.keys())}")
    if args.limit is not None:
        recordings = dict(list(recordings.items())[:args.limit])
    output = args.output or args.path / f"results-{args.model}-evaluated.jsonl"
    protected = [args.path / name for name in
                 ("manifest.jsonl", args.results_file, "results-neutral-binary.jsonl")]
    if output.resolve() in [p.resolve() for p in protected]:
        parser.error("Use a separate prediction output, not an existing input file")
    predictions = {r["clip_id"]: r for r in read_rows(output)} if output.exists() else {}
    adapter = importlib.import_module(f"{args.model}_agreement")
    if any(r["model"] != adapter.MODEL for r in predictions.values()):
        parser.error("Prediction file contains another model; choose a separate --output")
    # Reuse model-identical neutral/angry predictions from earlier agreement runs.
    # Reuse does not restrict inclusion: all other evaluated styles are inferred below.
    reused = []
    for style in ("neutral", "angry"):
        source = args.path / f"results-{args.model}-{style}.jsonl"
        if not source.exists():
            continue
        previous = {r["clip_id"]: r for r in read_rows(source)}
        for r in previous.values():
            k = r["clip_id"]
            if k in recordings and k not in predictions and r["model"] == adapter.MODEL:
                predictions[k] = r
                reused.append(r)
    if reused and not args.summary_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a") as stream:
            for row in reused:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    pending = [r for k, r in recordings.items() if k not in predictions]
    print(f"Selected {len(recordings)} evaluated clips across all generated styles; "
          f"reused {len(reused)} previous predictions; {len(pending)} pending.", flush=True)
    if pending and not args.summary_only:
        import librosa
        import soundfile as sf
        from emotion_agreement import binary_label

        predict = adapter.load_predictor(args.device)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a") as stream:
            for index, recording in enumerate(pending, 1):
                audio, sample_rate = sf.read(args.path / recording["audio_path"], dtype="float32")
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                if sample_rate != 16000:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                prediction = predict(audio)
                row = {"clip_id": recording["clip_id"], "question_id": recording["question_id"],
                       "style": recording["style"], "model": adapter.MODEL, **prediction}
                row["binary_prediction"] = binary_label(row["emotion"])
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                predictions[row["clip_id"]] = row
                print(f'[{index}/{len(pending)}] {row["clip_id"]}: {row["emotion"]}', flush=True)
    selected_predictions = {k: r for k, r in predictions.items() if k in recordings}
    summary = {"model": adapter.MODEL, "selected": len(recordings),
               "classified": len(selected_predictions),
               "missing_predictions": len(recordings)-len(selected_predictions),
               "accuracy_groups": accuracy_rows(selected_predictions, answers)}
    print_accuracy(summary["accuracy_groups"], args.by_style)
    summary_path = output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nPredictions: {output}\nSummary (including per-style groups and CIs): {summary_path}")


if __name__ == "__main__":
    main()
