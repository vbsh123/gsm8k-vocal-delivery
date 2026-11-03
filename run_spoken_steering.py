"""DIFFA-classify original Spoken GSM, then test fixed L4 acoustic x1 on its neutral subset."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

from emotion_agreement import read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("outputs/spoken-gsm-original"))
    parser.add_argument("--vectors", type=Path, default=Path("outputs/manual200-mixed/style_vectors.pt"))
    parser.add_argument("--num-examples", type=int, default=300, help="Number of source clips to classify, before selecting neutral")
    parser.add_argument("--output", type=Path, default=Path("outputs/steer-spoken-gsm-diffa-neutral-L4-acoustic-a1"))
    parser.add_argument("--prepare-only", action="store_true", help="Inspect existing classifications; do not run inference")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    for name in ("path", "vectors", "output"):
        setattr(args, name, (repo / getattr(args, name)).resolve())
    if args.num_examples < 1:
        parser.error("--num-examples must be positive")
    if args.path == args.output:
        parser.error("Use a separate steering output folder")
    if not (args.path / "manifest.jsonl").exists() or not args.vectors.exists():
        parser.error("Original Spoken GSM manifest and the existing vector file must exist")
    records = read_rows(args.path / "manifest.jsonl")
    if len(records) < args.num_examples:
        parser.error(f"Only {len(records)} original clips prepared; run prepare_spoken_gsm.py --num-examples {args.num_examples}")
    cohort = dict(list(records.items())[:args.num_examples])
    if any(r["style"] != "original" or not 0 < r["duration_seconds"] <= 30 for r in cohort.values()):
        parser.error("Expected original Spoken GSM recordings within 30 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    selection_file = args.output / "selection.json"
    exclusions_file = args.output / "excluded-questions.jsonl"
    if selection_file.exists():
        plan = json.loads(selection_file.read_text())
        if plan["source_count"] != args.num_examples or plan["source_path"] != str(args.path):
            parser.error("Source cohort changed; use a new --output")
        if not exclusions_file.exists():
            parser.error("Frozen selection is missing its exclusions file")
        selected = plan["recordings"]
        print(f"Reusing {len(selected)} frozen DIFFA-neutral original recordings.", flush=True)
    else:
        if not args.prepare_only:
            subprocess.run([sys.executable, "-u", str(repo / "classify_spoken_gsm.py"),
                            "--path", str(args.path), "--model", "diffa",
                            "--limit", str(args.num_examples)], cwd=repo, check=True)
        labels = read_rows(args.path / "results-diffa-original.jsonl")
        missing = cohort.keys() - labels.keys()
        if missing:
            print(f"{len(missing)}/{len(cohort)} original clips still need DIFFA classification.")
            print("Run without --prepare-only to classify them, then steer the neutral subset.")
            return
        counts = Counter(labels[k]["predicted_style"] for k in cohort)
        print(f"DIFFA labels among {len(cohort)} originals: {dict(counts)}", flush=True)
        selected = [r for k, r in cohort.items() if labels[k]["predicted_style"] == "yes"]
        if not selected:
            print("No DIFFA-neutral original clips. Steering was not started.")
            return
        plan = {"recordings": selected, "source_path": str(args.path),
                "source_count": args.num_examples, "selection_rule": "original audio; DIFFA predicted yes",
                "question_overlap": "Not verified: source audio rows are not mapped 1:1 to text GSM8K IDs"}
        selection_file.write_text(json.dumps(plan, indent=2) + "\n")
        # These source-audio IDs cannot be matched to the text-question exclusion
        # IDs. This is an audio-source transfer test, not a guaranteed question holdout.
        exclusions_file.write_text("")
    print(f"{len(selected)} original questions x 2 conditions: baseline and L4 acoustic x1.", flush=True)
    if not args.prepare_only:
        subprocess.run([sys.executable, "-u", str(repo / "steer.py"),
                        "--path", str(args.path), "--vectors", str(args.vectors),
                        "--exclude-results", str(exclusions_file),
                        "--questions-from", str(selection_file), "--output", str(args.output),
                        "--num-examples", str(len(selected)), "--style", "angry",
                        "--layers", "4", "--groups", "acoustic", "--strengths", "0", "1",
                        "--mode", "fixed"], cwd=repo, check=True)


if __name__ == "__main__":
    main()
