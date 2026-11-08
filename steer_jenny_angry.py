"""Run the existing steering engine on held-out Jenny neutral recordings."""

import argparse
import json
from pathlib import Path
import random
import sys
import wave

import steer

from compare_cloud_accuracy import read
from extract_jenny_angry import freeze


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=Path("outputs/azure-jenny-next500"))
    p.add_argument("--vectors", type=Path, default=Path("outputs/jenny-angry-vectors/style-vectors.pt"))
    p.add_argument("--output", type=Path, default=Path("outputs/steer-jenny-angry-L0-25-n50"))
    p.add_argument("--num-examples", type=int, default=50, help="Use 0 for all eligible held-out neutrals.")
    p.add_argument("--layers", type=int, nargs="+", default=list(range(26)))
    p.add_argument("--groups", nargs="+", choices=["acoustic", "semantic", "combined", "answer"],
                   default=["acoustic", "answer"])
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-runs", type=Path, nargs="+", default=[],
                   help="Exclude question IDs in previous steering run_config.json files.")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args(argv)
    excluded_path = args.vectors.parent / "extraction-questions.jsonl"
    training_questions = {r["question_id"] for r in read(excluded_path)}
    previously_tested = {r["question_id"] for path in args.exclude_runs
                         for r in json.loads(path.read_text())["recordings"]}
    excluded = training_questions | previously_tested
    selected = []
    for task in read(args.path / "plan.jsonl"):
        if task["provider"] != "azure" or task["style"] != "neutral":
            continue
        if task["question_id"] in training_questions:
            raise ValueError(f'Extraction/test question overlap: {task["question_id"]}')
        if task["question_id"] in previously_tested:
            continue
        audio_path = (args.path / task["audio_path"]).resolve()
        with wave.open(str(audio_path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        if duration > 30:
            continue
        selected.append({"clip_id": f'{task["question_id"]}__{task["voice"]}__neutral__r00',
                         "question_id": task["question_id"], "style": "neutral",
                         "speaker": task["voice"], "repeat": 0, "audio_path": str(audio_path),
                         "duration_seconds": duration,
                         "answer": str(task["answer"]).rsplit("####", 1)[-1].strip()})
    selected.sort(key=lambda row: row["question_id"])
    random.Random(args.seed).shuffle(selected)
    if args.num_examples:
        selected = selected[:args.num_examples]
    inputs = args.output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    if args.exclude_runs:
        excluded_path = inputs / "excluded-questions.jsonl"
        freeze(excluded_path, [{"question_id": qid} for qid in sorted(excluded)])
        print(f"Excluded {len(previously_tested)} previously tested question IDs.", flush=True)
    freeze(inputs / "manifest.jsonl", selected)
    question_path = inputs / "questions.json"
    question_plan = {"recordings": selected}
    if question_path.exists() and json.loads(question_path.read_text()) != question_plan:
        p.error("Saved test selection changed; use another --output")
    question_path.write_text(json.dumps(question_plan, indent=2) + "\n")
    conditions = 1 + len(args.layers) * len(args.groups)
    print(f"{len(selected)} held-out Jenny neutrals; {conditions} conditions; "
          f"{len(selected)*conditions} generations before resume.", flush=True)
    print("Test selection uses neutral TTS style, without filtering by classifier or correctness.", flush=True)

    command = ["steer.py", "--path", str(inputs), "--vectors", str(args.vectors),
               "--exclude-results", str(excluded_path), "--questions-from", str(question_path),
               "--output", str(args.output), "--num-examples", str(len(selected)),
               "--style", "angry", "--mode", "fixed", "--strengths", str(args.strength),
               "--layers", *map(str, args.layers), "--groups", *args.groups, "--seed", str(args.seed)]
    if args.prepare_only:
        command.append("--prepare-only")
    previous = sys.argv
    try:
        sys.argv = command
        steer.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
