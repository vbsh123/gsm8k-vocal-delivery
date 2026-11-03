"""Run L4 acoustic x1 on remaining DIFFA-neutral questions. No edits to steer.py."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from analyze_steering_groups import read_rows


def select_remaining(records, labels, answers, extracted, tested):
    passing = {r["clip_id"] for r in labels if r["style"] == "neutral" and r["predicted_style"] == "yes"}
    evaluated = {r["clip_id"] for r in answers}
    candidates = {r["question_id"]: r for r in records
                  if r["clip_id"] in passing
                  and r["style"] == "neutral" and r["speaker"] == "Ryan" and r["repeat"] == 0
                  and r["duration_seconds"] <= 30}
    selected = [r for q, r in sorted(candidates.items()) if q not in extracted and q not in tested]
    return selected, {"diffa_neutral_clips": len(passing),
                      "eligible_questions": len(candidates),
                      "with_saved_math_answer": sum(r["clip_id"] in evaluated for r in candidates.values()),
                      "already_tested": sum(q in tested for q in candidates),
                      "used_for_vector_extraction": sum(q in extracted for q in candidates),
                      "remaining": len(selected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("outputs/manual200-mixed/cropped"))
    parser.add_argument("--vectors", type=Path, default=Path("outputs/manual200-mixed/style_vectors.pt"))
    parser.add_argument("--previous-run", type=Path, default=Path("outputs/steer-angry-L0-15-dev200-a1"))
    parser.add_argument("--extraction-questions", type=Path,
                        default=Path("outputs/manual200-mixed/vector-extraction-questions.jsonl"))
    parser.add_argument("--pilot-results", type=Path, default=Path("outputs/steer-angry-all32-dev/results.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/steer-diffa-neutral-new-L4-acoustic-a1"))
    parser.add_argument("--prepare-only", action="store_true", help="Count and freeze selection without inference")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    for name in ("path", "vectors", "previous_run", "extraction_questions", "pilot_results", "output"):
        setattr(args, name, (repo / getattr(args, name)).resolve())
    for path in [args.path / "manifest.jsonl", args.path / "results-neutral-binary.jsonl",
                 args.extraction_questions,
                 args.previous_run / "run_config.json", args.pilot_results, args.vectors]:
        if not path.exists():
            parser.error(f"Missing required file: {path}")
    args.output.mkdir(parents=True, exist_ok=True)
    selection_file = args.output / "selection.json"
    exclusions_file = args.output / "excluded-questions.jsonl"
    if selection_file.exists():
        if not exclusions_file.exists():
            parser.error("Frozen selection is missing its exclusions file")
        plan = json.loads(selection_file.read_text())
        selected, counts = plan["recordings"], plan["selection_counts"]
        print("Reusing the frozen question selection.")
    else:
        extracted = {r["question_id"] for r in read_rows(args.extraction_questions)}
        previous = json.loads((args.previous_run / "run_config.json").read_text())["recordings"]
        tested = {r["question_id"] for r in previous}
        tested.update(r["question_id"] for r in read_rows(args.pilot_results))
        labels = {r["clip_id"]: r for r in read_rows(args.path / "results-neutral-binary.jsonl")}
        answer_file = args.path / "results-reasoning-v2.jsonl"
        answers = read_rows(answer_file) if answer_file.exists() else []
        selected, counts = select_remaining(read_rows(args.path / "manifest.jsonl"), list(labels.values()),
                                            answers, extracted, tested)
        if selected:
            selection_file.write_text(json.dumps({"recordings": selected, "selection_counts": counts}, indent=2) + "\n")
            exclusions_file.write_text("".join(json.dumps({"question_id": q}) + "\n"
                                               for q in sorted(extracted | tested)))
    print(json.dumps(counts, indent=2), flush=True)
    if not selected:
        print("No remaining eligible questions. No model was loaded.")
        return
    command = [sys.executable, str(repo / "steer.py"), "--path", str(args.path),
               "--vectors", str(args.vectors), "--exclude-results", str(exclusions_file),
               "--questions-from", str(selection_file), "--output", str(args.output),
               "--num-examples", str(len(selected)), "--style", "angry", "--layers", "4",
               "--groups", "acoustic", "--strengths", "0", "1", "--mode", "fixed"]
    print(f"\n{len(selected)} questions x 2 conditions: baseline and L4 acoustic x1.", flush=True)
    if not args.prepare_only:
        subprocess.run(command, cwd=repo, check=True)


if __name__ == "__main__":
    main()
