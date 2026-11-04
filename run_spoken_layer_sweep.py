"""Run other layers on the same original Spoken GSM clips, reusing validated baselines."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scoring import score_response
from steer import make_conditions, read_rows


def source_baselines(plan, results):
    settings = plan["settings"]
    selected = plan["recordings"]
    latest = {r["clip_id"]: r for r in results if r["condition"] == "baseline"}
    baselines = []
    for recording in selected:
        clip_id = recording["clip_id"]
        if clip_id not in latest:
            raise ValueError(f"Source baseline missing for {clip_id}")
        row = latest[clip_id]
        seed = int.from_bytes(hashlib.sha256(f"{settings['seed']}:{clip_id}".encode()).digest()[:4], "big")
        if (row["seed"] != seed or row["steered_forwards"] != 0 or row["strength"] != 0
                or row["forwards"] <= 0 or row["question_id"] != recording["question_id"]
                or row["reference_answer"] != str(recording["answer"])):
            raise ValueError(f"Source baseline metadata mismatch for {clip_id}")
        score = score_response(row["response"], recording["answer"])
        if any(row.get(k) != v for k, v in score.items()):
            raise ValueError(f"Source baseline scoring mismatch for {clip_id}")
        baselines.append(row)
    if len({(r["prompt"], r["max_new_tokens"]) for r in baselines}) != 1:
        raise ValueError("Source baselines use inconsistent generation settings")
    return baselines


def import_baselines(path, baselines, selected, conditions):
    existing = {(r["clip_id"], r["condition"]): r for r in read_rows(path)}
    allowed_ids = {r["clip_id"] for r in selected}
    allowed_conditions = {r["name"] for r in conditions}
    if any(k not in allowed_ids or c not in allowed_conditions for k, c in existing):
        raise ValueError("Output contains results from another cohort or condition")
    copied = 0
    with path.open("a") as stream:
        for row in baselines:
            key = (row["clip_id"], "baseline")
            if key in existing:
                if existing[key] != row:
                    raise ValueError(f"Existing baseline differs from source: {row['clip_id']}")
                continue
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            existing[key] = row
            copied += 1
    return copied, existing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path,
                        default=Path("outputs/steer-spoken-gsm-diffa-neutral-L4-acoustic-a1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/steer-spoken-gsm-L15-25-a1"))
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(15, 26)))
    parser.add_argument("--groups", nargs="+", choices=("acoustic", "answer"), default=["acoustic", "answer"])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    args.source_run = (repo / args.source_run).resolve()
    args.output = (repo / args.output).resolve()
    if args.output == args.source_run:
        parser.error("Use a separate output folder")
    if any(layer < 0 or layer >= 32 for layer in args.layers):
        parser.error("Layers must be between 0 and 31")
    source_config = args.source_run / "run_config.json"
    plan = json.loads(source_config.read_text())
    settings, selected = plan["settings"], plan["recordings"]
    if not selected or settings["style"] != "angry" or settings["mode"] != "fixed":
        parser.error("Expected the completed fixed angry-direction source run")
    vectors = Path(settings["vectors"])
    stat = vectors.stat()
    if stat.st_size != settings["vector_bytes"] or stat.st_mtime_ns != settings["vector_mtime_ns"]:
        parser.error("Vector file changed since the source run")
    try:
        conditions = make_conditions(args.layers, args.groups, [0, 1])
        baselines = source_baselines(plan, read_rows(args.source_run / "results.jsonl"))
    except ValueError as exc:
        parser.error(str(exc))
    command = [sys.executable, "-u", str(repo / "steer.py"),
               "--path", settings["path"], "--vectors", str(vectors),
               "--exclude-results", *settings["exclude_results"],
               "--questions-from", str(source_config), "--output", str(args.output),
               "--num-examples", str(len(selected)), "--seed", str(settings["seed"]),
               "--style", "angry", "--mode", "fixed", "--min-pairs", str(settings["min_pairs"]),
               "--layers", *map(str, args.layers), "--groups", *args.groups, "--strengths", "0", "1"]
    # Let steer.py validate/freeze the exact plan before importing baseline rows.
    prepared = subprocess.run(command + ["--prepare-only"], cwd=repo, capture_output=True, text=True)
    if prepared.returncode:
        print(prepared.stdout, end="")
        print(prepared.stderr, end="", file=sys.stderr)
        prepared.check_returncode()
    try:
        copied, existing = import_baselines(args.output / "results.jsonl", baselines, selected, conditions)
    except ValueError as exc:
        parser.error(str(exc))
    pending = sum((r["clip_id"], c["name"]) not in existing for r in selected for c in conditions[1:])
    print(f"{len(selected)} source clips; {len(conditions)-1} separate steering conditions.", flush=True)
    print(f"Reused {len(baselines)} baselines ({copied} copied now); {pending} new generations remain.", flush=True)
    if not args.prepare_only:
        subprocess.run(command, cwd=repo, check=True)


if __name__ == "__main__":
    main()
