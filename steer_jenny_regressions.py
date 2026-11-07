"""Check angry steering regressions where saved neutral and angry answers were both correct."""

import argparse
import json
from pathlib import Path
import wave

from steer import read_rows
from steer_jenny_recovery import correct
from steer_jenny_six import main as run_diagnostic


TARGETS = [(11, "answer", 1), (12, "acoustic", 1), (11, "answer", 1.5), (0, "answer", 1)]


def select_regressions(tasks, answers, excluded):
    clips = {(r["question_id"], r["voice"], r["style"]): r
             for r in tasks if r["provider"] == "azure"}
    selected = []
    for key, neutral in sorted(clips.items()):
        if key[2] != "neutral" or key[0] in excluded:
            continue
        angry = clips.get((key[0], key[1], "angry"))
        if angry is None or any(r["audio_path"] not in answers for r in (neutral, angry)):
            continue
        if all(correct(answers[r["audio_path"]]) for r in (neutral, angry)):
            selected.append(neutral)
    return selected


def regression_summary(output, details=False):
    config = json.loads((output / "run_config.json").read_text())
    latest = {(r["clip_id"], r["condition"]): r for r in read_rows(output / "results.jsonl")}
    print("\nRegressions among fresh-baseline successes (selected diagnostic)")
    print(f'{"Condition":24} {"Paired":>7} {"Fresh correct":>14} {"Lost":>7} {"Loss rate":>11}')
    for c in config["settings"]["conditions"][1:]:
        pairs = [(r, latest[r["clip_id"], "baseline"], latest[r["clip_id"], c["name"]])
                 for r in config["recordings"]
                 if (r["clip_id"], "baseline") in latest and (r["clip_id"], c["name"]) in latest]
        successes = [(r, b, s) for r, b, s in pairs if correct(b)]
        lost = [r["question_id"] for r, b, s in successes if not correct(s)]
        rate = f'{len(lost)/len(successes):.1%}' if successes else "n/a"
        print(f'{c["name"]:24} {len(pairs):7} {len(successes):14} {len(lost):7} {rate:>11}')
        if details:
            print("  Lost questions: " + (", ".join(lost) or "none"))
    baselines = [latest[r["clip_id"], "baseline"] for r in config["recordings"]
                 if (r["clip_id"], "baseline") in latest]
    print(f"Fresh baseline failures excluded from loss rate: {sum(not correct(r) for r in baselines)}"
          f"/{len(baselines)} completed baselines.")
    print("Selected from both-correct original answers; this is not an unselected accuracy estimate.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=Path("outputs/azure-jenny-next500"))
    p.add_argument("--previous", type=Path, default=Path("outputs/steer-jenny-angry-L0-25-n50"))
    p.add_argument("--vectors", type=Path, default=Path("outputs/jenny-angry-vectors/style-vectors.pt"))
    p.add_argument("--output", type=Path, default=Path("outputs/steer-jenny-angry-regressions"))
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--details", action="store_true", help="Print regressed question IDs in the summary.")
    args = p.parse_args(argv)
    if args.summary_only:
        regression_summary(args.output, args.details)
        return
    excluded = {r["question_id"] for r in read_rows(args.vectors.parent / "extraction-questions.jsonl")}
    answers = {r["audio_path"]: r for r in read_rows(args.path / "results-matched-accuracy.jsonl")}
    selected = select_regressions(read_rows(args.path / "plan.jsonl"), answers, excluded)
    recordings = []
    for task in selected:
        audio_path = (args.path / task["audio_path"]).resolve()
        with wave.open(str(audio_path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        if not 0 < duration <= 30:
            continue
        recordings.append({"clip_id": f'{task["question_id"]}__{task["voice"]}__neutral__r00',
                           "question_id": task["question_id"], "style": "neutral",
                           "speaker": task["voice"], "repeat": 0, "duration_seconds": duration,
                           "audio_path": str(audio_path),
                           "answer": str(task["answer"]).rsplit("####", 1)[-1].strip()})
    if not recordings:
        p.error("No eligible both-correct questions")
    conditions = [{"name": "baseline", "interventions": []}] + [
        {"name": f"L{layer}-{group}-a{strength:g}", "interventions": [
            {"layer": layer, "group": group, "strength": strength}]}
        for layer, group, strength in TARGETS]
    prepared = {"recordings": recordings, "conditions": conditions,
                "selection": "original neutral and angry both correct; excludes vector extraction questions",
                "source": str(args.path.resolve()), "excluded_question_ids": sorted(excluded)}
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "selected-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != prepared:
        p.error("Saved regression selection changed; use a new --output")
    plan_path.write_text(json.dumps(prepared, indent=2) + "\n")
    print(f"{len(recordings)} both-correct questions; four steering conditions plus one shared baseline.", flush=True)
    command = ["--previous", str(args.previous), "--vectors", str(args.vectors),
               "--output", str(args.output), "--prepared-plan", str(plan_path)]
    if args.prepare_only:
        command.append("--prepare-only")
    run_diagnostic(command)
    if not args.prepare_only:
        regression_summary(args.output, args.details)


if __name__ == "__main__":
    main()
