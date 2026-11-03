"""Rank saved steering conditions within original-input classifier groups; no inference."""

import argparse
import csv
from pathlib import Path

from analyze_steering_groups import flexible_correct, paired_stats, read_rows


DEFAULT_RUNS = ["steer-angry-L0-15-dev200-a1", "steer-angry-L15-25-dev200-a1",
                "steer-angry-followup200-sem15-25", "steer-angry-higher-strength200"]


def classifier_groups(ids, labels, is_diffa=False):
    field = "predicted_style" if is_diffa else "binary_prediction"
    mapping = {"yes": "neutral", "no": "emotional"} if is_diffa else {
        "neutral": "neutral", "emotional": "emotional", "unresolved": "unresolved"}
    groups = {}
    for k in ids:
        row = labels.get(k, {})
        group = mapping.get(row.get(field), "missing/invalid")
        groups.setdefault(("binary", group), []).append(k)
        if not is_diffa and row:
            groups.setdefault(("emotion", row["emotion"]), []).append(k)
    return groups


def analyze_run(rows, sources, scoring, run_name):
    # Keep each run's own baseline; never mix different plans or rerun outputs.
    baseline = {r["clip_id"]: r for r in rows if r["condition"] == "baseline"}
    conditions = {}
    for row in rows:
        if row["condition"] != "baseline":
            conditions.setdefault(row["condition"], {})[row["clip_id"]] = row
    scores = {"strict": lambda r: r["correct"], "non-strict": flexible_correct}
    output = []
    for condition, steered in conditions.items():
        ids = sorted(baseline.keys() & steered.keys())
        for model, labels in sources.items():
            groups = classifier_groups(ids, labels, model == "diffa")
            for (kind, group), selected in groups.items():
                for mode, score in scores.items():
                    if scoring != "both" and scoring != mode:
                        continue
                    stats = paired_stats(selected, baseline, steered, score)
                    if stats:
                        output.append({"run": run_name, "condition": condition, "classifier": model,
                                       "group_type": kind, "group": group, "scoring": mode,
                                       **stats, "baseline_pct": 100*stats["baseline"]/stats["n"],
                                       "steered_pct": 100*stats["steered"]/stats["n"]})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("outputs/manual200-mixed/cropped"))
    parser.add_argument("--results", nargs="+", type=Path, help="One or more steering results.jsonl files")
    parser.add_argument("--scoring", choices=("strict", "non-strict", "both"), default="both")
    parser.add_argument("--min-pairs", type=int, default=20, help="Minimum group size for printed rankings")
    parser.add_argument("--top", type=int, default=5, help="Best and worst conditions printed per binary group")
    parser.add_argument("--output", type=Path, default=Path("outputs/steering-group-effects.csv"))
    args = parser.parse_args()
    if args.top < 1 or args.min_pairs < 1:
        parser.error("--top and --min-pairs must be positive")
    runs = args.results or [Path("outputs") / r / "results.jsonl" for r in DEFAULT_RUNS
                            if (Path("outputs") / r / "results.jsonl").exists()]
    if not runs:
        parser.error("No steering runs found; provide --results")
    sources = {"diffa": {r["clip_id"]: r for r in read_rows(args.path / "results-neutral-binary.jsonl")}}
    for model in ("emotion2vec", "sensevoice", "speechbrain"):
        labels = {}
        for filename in (f"results-{model}-neutral.jsonl", f"results-{model}-evaluated.jsonl"):
            source = args.path / filename
            if source.exists():
                labels.update({r["clip_id"]: r for r in read_rows(source) if r["style"] == "neutral"})
        if labels:
            sources[model] = labels
        else:
            print(f"No {model} labels found; skipping this classifier.")
    output = []
    for path in runs:
        rows = read_rows(path)
        baseline_count = len({r["clip_id"] for r in rows if r["condition"] == "baseline"})
        print(f"Read {path}: {baseline_count} baseline clips")
        output.extend(analyze_run(rows, sources, args.scoring, path.parent.name))
    if not output:
        parser.error("No paired steering conditions found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    keys = sorted({(r["scoring"], r["classifier"], r["group"]) for r in output
                   if r["group_type"] == "binary" and r["group"] != "missing/invalid"})
    for mode, model, group in keys:
        candidates = [r for r in output if r["scoring"] == mode and r["classifier"] == model
                      and r["group_type"] == "binary" and r["group"] == group
                      and r["n"] >= args.min_pairs]
        if not candidates:
            continue
        print(f"\n{model} / {group} / {mode}")
        for title, reverse in (("Best gains", True), ("Worst gains", False)):
            ranked = sorted(candidates, key=lambda r: r["gain_points"], reverse=reverse)[:args.top]
            print(f"  {title}:")
            for r in ranked:
                print(f'    {r["condition"]:35} n={r["n"]:3} '
                      f'baseline={r["baseline_pct"]:.1f}% steered={r["steered_pct"]:.1f}% '
                      f'gain={r["gain_points"]:+.1f}pp W/L={r["wins"]}/{r["losses"]} '
                      f'p={r["mcnemar_p"]:.4f} ({r["run"]})')
    print(f"\nAll layer/condition effects, including individual emotion tags: {args.output}")
    print("Ranks use gain, not final accuracy. Baselines are matched within each run and group.")
    print("Group membership and question counts can differ across classifiers/runs.")
    print("Layer and subgroup searches are exploratory; p-values are unadjusted.")


if __name__ == "__main__":
    main()
