"""Expand the saved 74-question recovery run, reusing baselines and completed conditions."""

import argparse
import hashlib
import json
from pathlib import Path

from steer import read_rows
from archive.steering.steer_jenny_recovery import recovery_summary
from archive.steering.steer_jenny_six import main as run_diagnostic


TARGETS = [(0, "answer", 1), (2, "acoustic", 1), (3, "answer", 1),
           (8, "answer", 1), (9, "answer", 1), (12, "acoustic", 1),
           (14, "answer", 1), (23, "acoustic", 1), (11, "answer", 1.5)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, default=Path("outputs/steer-jenny-angry-recovery74"))
    p.add_argument("--output", type=Path, default=Path("outputs/steer-jenny-angry-recovery74-expanded"))
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args(argv)
    if args.summary_only:
        run_diagnostic(["--output", str(args.output), "--summary-only"])
        recovery_summary(args.output)
        return
    source = json.loads((args.source_run / "run_config.json").read_text())
    settings = source["settings"]
    vectors = Path(settings["vectors"])
    stat = vectors.stat()
    if stat.st_size != settings["vector_bytes"] or stat.st_mtime_ns != settings["vector_mtime_ns"]:
        p.error("Vector checkpoint changed since the source run; cached results cannot be reused")
    previous = Path(settings["previous"])
    if json.loads((previous / "run_config.json").read_text())["settings"]["seed"] != settings["seed"]:
        p.error("Source seed differs from the previous sweep")
    conditions = list(settings["conditions"])
    names = {c["name"] for c in conditions}
    added = []
    for layer, group, strength in TARGETS:
        name = f"L{layer}-{group}-a{strength:g}"
        if name not in names:
            conditions.append({"name": name, "interventions": [
                {"layer": layer, "group": group, "strength": strength}]})
            names.add(name)
            added.append(name)
    prepared = {"recordings": source["recordings"], "conditions": conditions,
                "source_run": str(args.source_run.resolve()),
                "selection": "same frozen recovery candidates; original sweep and training excluded"}
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "selected-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != prepared:
        p.error("Source selection/settings changed; use a new --output")
    plan_path.write_text(json.dumps(prepared, indent=2) + "\n")
    results_path = args.output / "results.jsonl"
    completed = {(r["clip_id"], r["condition"]) for r in read_rows(results_path)}
    selected_ids = {r["clip_id"] for r in source["recordings"]}
    copied = 0
    latest = {(r["clip_id"], r["condition"]): r for r in read_rows(args.source_run / "results.jsonl")}
    with results_path.open("a") as stream:
        for key, row in latest.items():
            if key in completed or key[0] not in selected_ids or key[1] not in names:
                continue
            expected_seed = int.from_bytes(hashlib.sha256(
                f'{settings["seed"]}:{row["clip_id"]}'.encode()).digest()[:4], "big")
            if row["seed"] != expected_seed:
                p.error("Cached result seed differs from the steering recipe")
            stream.write(json.dumps(row) + "\n")
            completed.add(key)
            copied += 1
    pending = sum((r["clip_id"], c["name"]) not in completed
                  for r in source["recordings"] for c in conditions)
    print(f"Reused {copied} source results; {pending} generations pending.", flush=True)
    print("Additional settings: " + ", ".join(added), flush=True)
    command = ["--previous", str(previous), "--vectors", str(vectors),
               "--output", str(args.output), "--prepared-plan", str(plan_path)]
    if args.prepare_only:
        command.append("--prepare-only")
    run_diagnostic(command)
    if not args.prepare_only:
        recovery_summary(args.output)


if __name__ == "__main__":
    main()
