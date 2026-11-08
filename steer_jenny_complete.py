"""Complete L11-answer/L12-acoustic x1 on all matched Jenny neutral/angry questions.

Reuse compatible answers from the recovery, regression, and original sweep;
generate only missing baseline/condition rows. Conditions remain separate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import wave

from scoring import score_response
from steer import read_rows, summarize
from steer_jenny_six import main as run_diagnostic


CONDITIONS = [{"name": "baseline", "interventions": []}] + [
    {"name": f"L{layer}-{group}-a1", "interventions": [
        {"layer": layer, "group": group, "strength": 1}]}
    for layer, group in [(11, "answer"), (12, "acoustic")]]


def cohort(path, excluded):
    tasks = {(r["question_id"], r["voice"], r["style"]): r
             for r in read_rows(path/"plan.jsonl") if r["provider"] == "azure"}
    answers = {r["audio_path"]: r for r in read_rows(path/"results-matched-accuracy.jsonl")}
    recordings = []
    for (qid, voice, style), neutral in sorted(tasks.items()):
        if style != "neutral" or qid in excluded:
            continue
        angry = tasks.get((qid, voice, "angry"))
        if angry is None or any(r["audio_path"] not in answers for r in (neutral, angry)):
            continue
        valid = True
        for task in (neutral, angry):
            with wave.open(str(path/task["audio_path"]), "rb") as handle:
                duration = handle.getnframes()/handle.getframerate()
            valid &= 0 < duration <= 30
        if not valid:
            continue
        with wave.open(str(path/neutral["audio_path"]), "rb") as handle:
            duration = handle.getnframes()/handle.getframerate()
        recordings.append(dict(clip_id=f"{qid}__{voice}__neutral__r00", question_id=qid,
            style="neutral", speaker=voice, repeat=0, duration_seconds=duration,
            audio_path=str((path/neutral["audio_path"]).resolve()),
            answer=str(neutral["answer"]).rsplit("####", 1)[-1].strip(),
            original_neutral_correct=score_response(answers[neutral["audio_path"]]["response"],
                                                   answers[neutral["audio_path"]]["reference_answer"])["correct"],
            original_angry_correct=score_response(answers[angry["audio_path"]]["response"],
                                                 answers[angry["audio_path"]]["reference_answer"])["correct"]))
    return recordings


def report(output):
    config = json.loads((output/"run_config.json").read_text())
    recordings = config["recordings"]
    ids = {r["clip_id"] for r in recordings}
    names = {c["name"] for c in CONDITIONS}
    latest = {(r["clip_id"], r["condition"]): {**r, **score_response(r["response"], r["reference_answer"])}
              for r in read_rows(output/"results.jsonl")
              if r["clip_id"] in ids and r["condition"] in names}
    print(f"\nCompleted generations: {len(latest)}/{len(recordings)*len(CONDITIONS)}")
    baselines = [r for (cid, name), r in latest.items() if name == "baseline"]
    if baselines:
        k = sum(r["correct"] for r in baselines)
        print(f"Fresh/reused neutral baseline: {k}/{len(baselines)} = {k/len(baselines):.1%}")
    summary = summarize(list(latest.values()), CONDITIONS)
    n = len(recordings)
    print(f"\nOriginal saved audio evaluations on all {n} matched questions:")
    for style in ("neutral", "angry"):
        k = sum(r[f"original_{style}_correct"] for r in recordings)
        print(f"  {style}: {k}/{n} = {k/n:.1%}")
    print("Steering comparisons use the same baseline recipe, not the original saved audio evaluation.")
    print("Full matched cohort, but layers were chosen using its subsets; results remain exploratory.")
    return summary


def merge_cache(runs, recordings, vectors, seed):
    records = {r["clip_id"]: r for r in recordings}
    expected = {c["name"]: c["interventions"] for c in CONDITIONS}
    cached, conflicts = {}, set()
    recipe = None
    stat = vectors.stat()
    for run in runs:
        config = json.loads((run/"run_config.json").read_text())
        settings = config["settings"]
        if (Path(settings["vectors"]).resolve() != vectors.resolve()
                or settings["vector_bytes"] != stat.st_size
                or settings["vector_mtime_ns"] != stat.st_mtime_ns
                or settings["seed"] != seed or settings["mode"] != "fixed"
                or settings["style"] != "angry"):
            raise ValueError(f"Incompatible cached vector/seed/steering recipe: {run}")
        source_records = {r["clip_id"]: r for r in config["recordings"]}
        conditions = {c["name"]: c for c in settings["conditions"]}
        latest = {(r["clip_id"], r["condition"]): r for r in read_rows(run/"results.jsonl")}
        for key, row in latest.items():
            cid, name = key
            if cid not in records or name not in expected:
                continue
            source = source_records[cid]
            target = records[cid]
            if any(source[k] != target[k] for k in ("question_id", "audio_path", "answer")):
                raise ValueError(f"Cached recording differs: {run}: {cid}")
            condition = conditions[name]
            interventions = condition.get("interventions")
            if interventions is None:
                interventions = [] if name == "baseline" else [
                    {k: condition[k] for k in ("layer", "group", "strength")}]
            expected_seed = int.from_bytes(hashlib.sha256(f"{seed}:{cid}".encode()).digest()[:4], "big")
            current_recipe = (row["prompt"], row["max_new_tokens"], row["decoder"], row["mode"], row["vector_style"])
            recipe = recipe or current_recipe
            if (interventions != expected[name] or row["seed"] != expected_seed
                    or current_recipe != recipe):
                raise ValueError(f"Cached generation recipe differs: {run}: {key}")
            if key in cached and cached[key]["response"] != row["response"]:
                conflicts.add(cid)
            cached.setdefault(key, {**row, "reused_from": str(run.resolve())})
    # If an old question has inconsistent repeated answers, rerun all three
    # conditions for that question rather than silently mixing baselines.
    return {key: row for key, row in cached.items() if key[0] not in conflicts}, conflicts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=Path("outputs/azure-jenny-next500"))
    p.add_argument("--previous", type=Path, default=Path("outputs/steer-jenny-angry-L0-25-n50"))
    p.add_argument("--vectors", type=Path, default=Path("outputs/jenny-angry-vectors/style-vectors.pt"))
    p.add_argument("--cached-runs", type=Path, nargs="+", default=[
        Path("outputs/steer-jenny-angry-regressions"),
        Path("outputs/steer-jenny-angry-recovery74-expanded"),
        Path("outputs/steer-jenny-angry-L0-25-n50")])
    p.add_argument("--output", type=Path, default=Path("outputs/steer-jenny-angry-complete"))
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args(argv)
    if args.summary_only:
        report(args.output)
        return
    excluded = {r["question_id"] for r in read_rows(args.vectors.parent/"extraction-questions.jsonl")}
    recordings = cohort(args.path, excluded)
    seed = json.loads((args.previous/"run_config.json").read_text())["settings"]["seed"]
    cached, conflicts = merge_cache(args.cached_runs, recordings, args.vectors, seed)
    prepared = dict(recordings=recordings, conditions=CONDITIONS,
                    selection="all evaluated matched neutral/angry questions, without correctness or emotion filtering")
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output/"selected-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != prepared:
        p.error("Saved cohort changed; use a different --output")
    plan_path.write_text(json.dumps(prepared, indent=2)+"\n")
    results_path = args.output/"results.jsonl"
    completed = {(r["clip_id"], r["condition"]) for r in read_rows(results_path)}
    copied = 0
    with results_path.open("a") as handle:
        for key, row in cached.items():
            if key not in completed:
                handle.write(json.dumps(row)+"\n")
                completed.add(key)
                copied += 1
    print(f"Reused {copied} saved answers; {len(recordings)*3-len(completed)} generations pending.", flush=True)
    if conflicts:
        print(f"Rerunning {len(conflicts)} questions with inconsistent cached answers.", flush=True)
    command = ["--previous", str(args.previous), "--vectors", str(args.vectors),
               "--output", str(args.output), "--prepared-plan", str(plan_path)]
    if args.prepare_only:
        command.append("--prepare-only")
    run_diagnostic(command)
    if not args.prepare_only:
        summary = report(args.output)
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
