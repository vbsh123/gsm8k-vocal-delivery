"""Shared neutral-clip selection and agreement reporting; no DIFFA model loading."""

import argparse
from collections import Counter
import json
from pathlib import Path


EMOTIONS = {"angry", "disgusted", "fearful", "happy", "sad", "surprised"}


def binary_label(emotion):
    if emotion == "neutral":
        return "neutral"
    if emotion in EMOTIONS:
        return "emotional"
    return "unresolved"


def read_rows(path):
    if not path.exists():
        return {}
    with path.open() as stream:
        return {r["clip_id"]: r for line in stream if line.strip()
                for r in [json.loads(line)]}


def agreement_summary(recordings, labels, predictions):
    matrix = {group: Counter() for group in ("neutral", "emotional")}
    for clip_id in recordings:
        if clip_id in predictions:
            group = "neutral" if labels[clip_id]["predicted_style"] == "yes" else "emotional"
            matrix[group][predictions[clip_id]["binary_prediction"]] += 1
    rows = {}
    for group in matrix:
        counts = matrix[group]
        resolved = counts["neutral"] + counts["emotional"]
        rows[group] = {
            "selected": sum((labels[k]["predicted_style"] == "yes") == (group == "neutral")
                            for k in recordings),
            "processed": sum(counts.values()),
            "external_neutral": counts["neutral"],
            "external_emotional": counts["emotional"],
            "unresolved": counts["unresolved"],
            "agreement": counts[group] / resolved if resolved else None,
        }
    n = sum(matrix[g][p] for g in matrix for p in ("neutral", "emotional"))
    observed = sum(matrix[g][g] for g in matrix) / n if n else None
    expected = sum(
        sum(matrix[g][p] for p in ("neutral", "emotional")) *
        sum(matrix[h][g] for h in matrix) for g in matrix
    ) / n**2 if n else None
    kappa = (observed - expected) / (1 - expected) if n and expected < 1 else None
    return {"groups": rows, "resolved_pairs": n, "agreement": observed, "cohen_kappa": kappa}


def print_summary(summary):
    print("\nDIFFA group          Selected Processed External-neutral External-emotional Unresolved Agreement")
    for group, row in summary["groups"].items():
        agreement = f'{row["agreement"]:.1%}' if row["agreement"] is not None else "n/a"
        print(f'{group:20} {row["selected"]:8} {row["processed"]:9} '
              f'{row["external_neutral"]:16} {row["external_emotional"]:18} '
              f'{row["unresolved"]:10} {agreement:>9}')
    agreement = summary["agreement"]
    kappa = summary["cohen_kappa"]
    print("Overall agreement:", f"{agreement:.1%}" if agreement is not None else "n/a")
    print("Cohen's kappa:", f"{kappa:.4f}" if kappa is not None else "n/a")
    print("Agreement and kappa exclude unresolved predictions; DIFFA is a comparator, not ground truth.")


def run(name, model_id, load_predictor):
    parser = argparse.ArgumentParser(description=f"Compare {name} with DIFFA on neutral recordings only.")
    parser.add_argument("--path", type=Path, default=Path("outputs/manual200-mixed/cropped"))
    parser.add_argument("--labels-file", default="results-neutral-binary.jsonl")
    parser.add_argument("--output", type=Path, help="Prediction JSONL; defaults inside --path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, help="Maximum selected clips, for a smoke run")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    labels_path = args.path / args.labels_file
    if not labels_path.exists() or not (args.path / "manifest.jsonl").exists():
        parser.error("Both manifest.jsonl and the DIFFA labels file must exist")
    labels = read_rows(labels_path)
    recordings = {
        k: r for k, r in read_rows(args.path / "manifest.jsonl").items()
        if r["style"] == "neutral" and labels.get(k, {}).get("predicted_style") in ("yes", "no")
    }
    if args.limit is not None:
        recordings = dict(list(recordings.items())[:args.limit])
    output = args.output or args.path / f"results-{name}-neutral.jsonl"
    if output.resolve() in (labels_path.resolve(), (args.path / "manifest.jsonl").resolve()):
        parser.error("Output must be separate from the input manifest and DIFFA labels")
    predictions = read_rows(output)
    if any(r["model"] != model_id for r in predictions.values()):
        parser.error("Output contains a different model; use a separate --output")
    pending = [r for k, r in recordings.items() if k not in predictions]
    print(f"Selected {len(recordings)} neutral recordings; {len(pending)} pending.", flush=True)
    if pending and not args.summary_only:
        predict = load_predictor(args.device)
        import librosa
        import soundfile as sf

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a") as stream:
            for index, recording in enumerate(pending, 1):
                audio, sample_rate = sf.read(args.path / recording["audio_path"], dtype="float32")
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                if sample_rate != 16000:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                prediction = predict(audio)
                clip_id = recording["clip_id"]
                row = {"clip_id": clip_id, "question_id": recording["question_id"],
                       "style": "neutral", "model": model_id,
                       "diffa_prediction": labels[clip_id]["predicted_style"],
                       **prediction}
                row["binary_prediction"] = binary_label(row["emotion"])
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                predictions[clip_id] = row
                print(f'[{index}/{len(pending)}] {clip_id} DIFFA={row["diffa_prediction"]} '
                      f'{name}={row["emotion"]} ({row["binary_prediction"]})', flush=True)
    summary = agreement_summary(recordings, labels, predictions)
    print_summary(summary)
    if output.exists():
        summary_path = output.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Predictions: {output}\nSummary: {summary_path}")
