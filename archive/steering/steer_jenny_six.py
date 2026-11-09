"""Outcome-selected six-question diagnostic of angry steering strengths and combinations."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

from steer_audio import run_experiment
from scoring import score_response
from steer import read_rows, summarize


QUESTIONS = ["gsm8k_test_00708", "gsm8k_test_00736", "gsm8k_test_00746",
             "gsm8k_test_00781", "gsm8k_test_00792", "gsm8k_test_00987"]


def build_conditions(config, results, questions, plan="original"):
    latest = {(r["question_id"], r["condition"]): r for r in results}

    def correct(row):
        return score_response(row["response"], row["reference_answer"])["correct"]

    for qid in questions:
        baseline = latest.get((qid, "baseline"))
    recovered = set()
    for c in config["settings"]["conditions"]:
        if c["name"] == "baseline":
            continue
        if any((qid, c["name"]) in latest and correct(latest[qid, c["name"]]) for qid in questions):
            recovered.add((c["layer"], c["group"]))

    def condition(name, targets):
        return {"name": name, "interventions": [
            {"layer": layer, "group": group, "strength": strength} for layer, group, strength in targets]}

    if plan == "lower-strength":
        conditions = [condition("baseline", []),
                      condition("L5ac1+L11ans1+L3ans2", [
                          (5, "acoustic", 1), (11, "answer", 1), (3, "answer", 2)])]
        excluded = {(5, "acoustic"), (11, "answer"), (3, "answer")}
        for layer, group in sorted(recovered - excluded):
            conditions.append(condition(f"L{layer}-{group}-a0.5", [(layer, group, 0.5)]))
        return conditions

    conditions = [condition("baseline", []), condition("L5-acoustic-a2", [(5, "acoustic", 2)])]
    for layer, group, short in [(0, "acoustic", "L0-ac"), (11, "answer", "L11-ans")]:
        for strength in (1, 0.5):
            conditions.append(condition(f"L5-ac1+{short}{strength:g}",
                              [(5, "acoustic", 1), (layer, group, strength)]))
    for layer, group in sorted(recovered - {(5, "acoustic")}):
        conditions.append(condition(f"L{layer}-{group}-a2", [(layer, group, 2)]))
    return conditions


def report(results, conditions, recordings):
    summary = summarize(results, conditions)
    latest = {(r["question_id"], r["condition"]): r for r in results}
    print("\nPer-question correct conditions (outcome-selected diagnostic):")
    for recording in recordings:
        qid = recording["question_id"]
        baseline = latest.get((qid, "baseline"))
        fixed = [c["name"] for c in conditions[1:] if latest.get((qid, c["name"]), {}).get("correct")]
        print(f'{qid}: fresh baseline={baseline["correct"] if baseline else "pending"}; '
              f'correct conditions: {", ".join(fixed) or "none yet"}')
    print("Outcome-selected diagnostic: these accuracies/p-values do not estimate general steering benefit.")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, default=Path("outputs/steer-jenny-angry-L0-25-n50"))
    p.add_argument("--vectors", type=Path, default=Path("outputs/jenny-angry-vectors/style-vectors.pt"))
    p.add_argument("--plan", choices=["original", "lower-strength"], default="original")
    p.add_argument("--prepared-plan", type=Path, help="Explicit recordings and intervention conditions from a diagnostic adapter.")
    p.add_argument("--output", type=Path)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--fixed-seed", type=int, choices=[42], default=42,
                   help="Compatibility option: generation seed is always hardcoded to 42.")
    p.add_argument("--allow-extraction-overlap", action="store_true",
                   help="Explicit in-sample diagnostic only; otherwise extraction overlap is rejected.")
    args = p.parse_args(argv)
    args.output = args.output or Path("outputs/steer-jenny-angry-six-" + (
        "half" if args.plan == "lower-strength" else "followup"))
    config_path = args.output / "run_config.json"
    results_path = args.output / "results.jsonl"
    if args.summary_only:
        saved = json.loads(config_path.read_text())
        report(read_rows(results_path), saved["settings"]["conditions"], saved["recordings"])
        return
    previous = json.loads((args.previous / "run_config.json").read_text())
    if args.prepared_plan:
        prepared = json.loads(args.prepared_plan.read_text())
        selected, conditions = prepared["recordings"], prepared["conditions"]
    else:
        records = {r["question_id"]: r for r in previous["recordings"]}
        selected = [records[qid] for qid in QUESTIONS]
        conditions = build_conditions(previous, read_rows(args.previous / "results.jsonl"), QUESTIONS, args.plan)
    selected_ids = {r["question_id"] for r in selected}
    excluded = {r["question_id"] for r in read_rows(args.vectors.parent / "extraction-questions.jsonl")}
    if excluded & selected_ids and not args.allow_extraction_overlap:
        p.error("The diagnostic overlaps vector-extraction questions")
    stat = args.vectors.stat()
    settings = {"previous": str(args.previous.resolve()), "vectors": str(args.vectors.resolve()),
                "vector_bytes": stat.st_size, "vector_mtime_ns": stat.st_mtime_ns,
                "seed": previous["settings"]["seed"], "conditions": conditions,
                "style": "angry", "mode": "fixed", "selection": "outcome-selected diagnostic"}
    settings.update(seed=42, seed_policy="constant_before_every_generation")
    if args.plan != "original":
        settings["plan"] = args.plan
    if args.allow_extraction_overlap:
        settings["allow_extraction_overlap"] = True
        print("IN-SAMPLE diagnostic: extraction-question overlap is explicitly allowed.", flush=True)
    if args.prepared_plan:
        settings.update(prepared_plan=str(args.prepared_plan.resolve()),
                        prepared_plan_sha256=hashlib.sha256(args.prepared_plan.read_bytes()).hexdigest())
    plan = {"settings": settings, "recordings": selected}
    args.output.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and json.loads(config_path.read_text()) != plan:
        p.error("Saved diagnostic settings changed; use a new --output")
    config_path.write_text(json.dumps(plan, indent=2) + "\n")
    results = read_rows(results_path)
    if any(r.get("seed") != 42 for r in results):
        p.error("Cached answers use a different seed; use a new --output")
    completed = {(r["clip_id"], r["condition"]) for r in results}
    pending = sum((r["clip_id"], c["name"]) not in completed for r in selected for c in conditions)
    print(f"{len(selected)} questions x {len(conditions)} conditions = "
          f"{len(selected)*len(conditions)} generations; {pending} pending.", flush=True)
    if args.prepared_plan:
        print("Explicit diagnostic targets; each condition runs separately against a fresh baseline.", flush=True)
    elif args.plan == "lower-strength":
        print("Combination: L5 acoustic x1 + L11 answer x1 + L3 answer x2. Other targets individually x0.5.", flush=True)
    else:
        print("Combinations: L5 acoustic + L0 acoustic, or L5 acoustic + L11 answer.", flush=True)
    for c in conditions:
        print(f'  {c["name"]}: {c["interventions"]}', flush=True)
    if args.prepare_only:
        return
    if not pending:
        report(results, conditions, selected)
        return

    return run_experiment(selected, conditions, args.vectors, args.output,
                          Path(previous["settings"].get("path", ".")),
                          args.allow_extraction_overlap, report)


if __name__ == "__main__":
    main()
