"""Test frozen angry steering candidates on remaining neutral-failed/angry-correct questions."""

import argparse
import json
from pathlib import Path
import wave

from scoring import score_response
from steer import read_rows
from steer_jenny_six import main as run_diagnostic


def correct(row):
    return score_response(row["response"], row["reference_answer"])["correct"]


def select_recovery(tasks, answers, excluded):
    clips = {(r["question_id"], r["voice"], r["style"]): r for r in tasks if r["provider"] == "azure"}
    candidates = []
    for key, neutral in sorted(clips.items()):
        if key[2] != "neutral":
            continue
        angry = clips.get((key[0], key[1], "angry"))
        if angry is None or any(r["audio_path"] not in answers for r in (neutral, angry)):
            continue
        if not correct(answers[neutral["audio_path"]]) and correct(answers[angry["audio_path"]]):
            candidates.append(neutral)
    return candidates, [r for r in candidates if r["question_id"] not in excluded]


def recovery_summary(output):
    config = json.loads((output / "run_config.json").read_text())
    latest = {(r["clip_id"], r["condition"]): r for r in read_rows(output / "results.jsonl")}
    print("\nRecovery among fresh-baseline failures (selected diagnostic)")
    print(f'{"Condition":24} {"Paired":>7} {"Still failed":>13} {"Recovered":>10} {"Recovery":>10}')
    for c in config["settings"]["conditions"][1:]:
        pairs = [(latest[r["clip_id"], "baseline"], latest[r["clip_id"], c["name"]])
                 for r in config["recordings"]
                 if (r["clip_id"], "baseline") in latest and (r["clip_id"], c["name"]) in latest]
        failures = [(b, s) for b, s in pairs if not correct(b)]
        recovered = sum(correct(s) for b, s in failures)
        rate = f'{recovered/len(failures):.1%}' if failures else "n/a"
        print(f'{c["name"]:24} {len(pairs):7} {len(failures):13} {recovered:10} {rate:>10}')
    print("Questions already correct under the fresh baseline are excluded from the recovery denominator.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=Path("outputs/azure-jenny-next500"))
    p.add_argument("--previous", type=Path, default=Path("outputs/steer-jenny-angry-L0-25-n50"))
    p.add_argument("--vectors", type=Path, default=Path("outputs/jenny-angry-vectors/style-vectors.pt"))
    p.add_argument("--output", type=Path, default=Path("outputs/steer-jenny-angry-recovery74"))
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args(argv)
    if args.summary_only:
        run_diagnostic(["--output", str(args.output), "--summary-only"])
        recovery_summary(args.output)
        return
    old = json.loads((args.previous / "run_config.json").read_text())
    tested = {r["question_id"] for r in old["recordings"]}
    training = {r["question_id"] for r in read_rows(args.vectors.parent / "extraction-questions.jsonl")}
    answers = {r["audio_path"]: r for r in read_rows(args.path / "results-matched-accuracy.jsonl")}
    candidates, selected = select_recovery(read_rows(args.path / "plan.jsonl"), answers, tested | training)
    recordings = []
    for task in selected:
        audio_path = (args.path / task["audio_path"]).resolve()
        with wave.open(str(audio_path), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        if duration > 30:
            continue
        recordings.append({"clip_id": f'{task["question_id"]}__{task["voice"]}__neutral__r00',
                           "question_id": task["question_id"], "style": "neutral",
                           "speaker": task["voice"], "repeat": 0, "duration_seconds": duration,
                           "audio_path": str(audio_path),
                           "answer": str(task["answer"]).rsplit("####", 1)[-1].strip()})
    if not recordings:
        p.error("No remaining eligible recovery candidates")
    conditions = [{"name": "baseline", "interventions": []}]
    for layer, group, strength in [(5, "acoustic", 1), (3, "answer", 2), (11, "answer", 1)]:
        conditions.append({"name": f"L{layer}-{group}-a{strength}", "interventions": [
            {"layer": layer, "group": group, "strength": strength}]})
    prepared = {"recordings": recordings, "conditions": conditions,
                "selection": "original neutral incorrect, angry correct; excludes prior sweep and extraction",
                "source": str(args.path.resolve()), "excluded_question_ids": sorted(tested | training)}
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "selected-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != prepared:
        p.error("Saved recovery selection changed; use a new --output")
    plan_path.write_text(json.dumps(prepared, indent=2) + "\n")
    print(f"{len(candidates)} original recovery candidates; {len(recordings)} remaining after exclusions/duration.", flush=True)
    command = ["--previous", str(args.previous), "--vectors", str(args.vectors),
               "--output", str(args.output), "--prepared-plan", str(plan_path)]
    if args.prepare_only:
        command.append("--prepare-only")
    run_diagnostic(command)
    if not args.prepare_only:
        recovery_summary(args.output)


if __name__ == "__main__":
    main()
