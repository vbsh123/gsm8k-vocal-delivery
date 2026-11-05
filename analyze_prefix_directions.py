"""Compare mean audio-token shift directions induced by emotional text prefixes."""

import argparse
import csv
import json
import math
from pathlib import Path


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def cosine(a, b):
    denominator = norm(a)*norm(b)
    return dot(a, b)/denominator if denominator else float("nan")


def correlation(a, b):
    if len(a) < 2:
        return float("nan")
    mean_a, mean_b = sum(a)/len(a), sum(b)/len(b)
    return cosine([x-mean_a for x in a], [y-mean_b for y in b])


def direction(entry, layer, group, mode):
    if mode == "first":
        count = int(entry["counts"][0])
        if not count:
            raise ValueError("No first-forward contributors")
        return (entry["sums"][layer, 0, group] / count).tolist(), count
    count = entry["pair_count"]
    if not count:
        raise ValueError("No question-pair contributors")
    return (entry["pair_mean_sum"][layer, group] / count).tolist(), count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vectors", type=Path, default=Path(
        "outputs/spoken-gsm-original/results-prompt-prefix.hidden-states.pt"))
    p.add_argument("--results", type=Path)
    p.add_argument("--layers", type=int, nargs="+", default=[15, 25, 31])
    p.add_argument("--groups", nargs="+", choices=["acoustic", "semantic"], default=["acoustic"])
    p.add_argument("--modes", nargs="+", choices=["first", "mean"], default=["first", "mean"])
    p.add_argument("--reference", default="angry")
    args = p.parse_args()
    import torch

    data = torch.load(args.vectors, map_location="cpu", weights_only=True, mmap=True)
    emotions = list(data["stats"])
    if args.reference not in emotions:
        p.error("Reference must be one of: " + ", ".join(emotions))
    results_path = args.results or args.vectors.parent / "results-prompt-prefix.jsonl"
    rows = {}
    with results_path.open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows[row["clip_id"], row["condition"]] = row
    ids = data["completed_ids"]
    if not ids or any((clip, condition) not in rows for clip in ids for condition in ["baseline", *emotions]):
        p.error("Saved responses do not cover the completed hidden-state cohort")
    accuracy = {condition: sum(rows[clip, condition]["correct"] for clip in ids)/len(ids)
                for condition in ["baseline", *emotions]}
    print(f"{len(ids)} completed questions; baseline accuracy={accuracy['baseline']:.1%}.")
    print("First = first-forward mean difference. Mean = average forwards within each pair, then across pairs.")
    print("Projection: baseline=0, reference emotion=1 along its mean-shift axis.")
    output = []
    for layer in args.layers:
        for group in args.groups:
            group_index = data["groups"].index(group)
            for mode in args.modes:
                directions, counts = {}, {}
                for emotion in emotions:
                    directions[emotion], counts[emotion] = direction(data["stats"][emotion], layer, group_index, mode)
                reference = directions[args.reference]
                reference_norm = norm(reference)
                if reference_norm == 0:
                    print(f"L{layer} {group} {mode}: zero reference direction; skipped")
                    continue
                print(f"\nL{layer} {group}, {mode}; reference={args.reference}")
                print(f"{'Emotion':12} {'N':>5} {'Accuracy':>9} {'Gain pp':>8} {'Cos(ref)':>10} "
                      f"{'Projection':>11} {'Norm/ref':>10} {'Off-axis/ref':>13}")
                projections = []
                for emotion in emotions:
                    vector = directions[emotion]
                    projection = dot(vector, reference)/(reference_norm**2)
                    off_axis = norm([v-projection*r for v, r in zip(vector, reference)])/reference_norm
                    gain = 100*(accuracy[emotion]-accuracy["baseline"])
                    row = {"layer": layer, "group": group, "mode": mode, "emotion": emotion,
                           "reference": args.reference, "pairs": counts[emotion],
                           "accuracy": accuracy[emotion], "gain_pp": gain,
                           "cosine_reference": cosine(vector, reference), "projection": projection,
                           "norm_ratio": norm(vector)/reference_norm, "off_axis_ratio": off_axis}
                    output.append(row)
                    projections.append(projection)
                    print(f"{emotion:12} {counts[emotion]:5} {accuracy[emotion]:9.1%} {gain:+8.1f} "
                          f"{row['cosine_reference']:10.4f} {projection:11.4f} "
                          f"{row['norm_ratio']:10.4f} {off_axis:13.4f}")
                r = correlation(projections, [accuracy[e] for e in emotions])
                print(f"Projection versus accuracy: Pearson r={r:+.3f} across {len(emotions)} emotions (exploratory).")
                print("Pairwise cosine between mean SHIFT vectors:")
                print(f"{'':12}" + "".join(f"{e:>11}" for e in emotions))
                for emotion in emotions:
                    print(f"{emotion:12}" + "".join(
                        f"{cosine(directions[emotion], directions[other]):11.4f}" for other in emotions))
    if output:
        output_path = args.vectors.parent / "prefix-direction-analysis.csv"
        with output_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=output[0].keys())
            writer.writeheader()
            writer.writerows(output)
        print("\nSaved:", output_path)
    print("These are aggregate directions, not evidence of consistent shifts across individual questions.")
    print("Later-forward means may also reflect diverging generated answers and unequal trajectory lengths.")


if __name__ == "__main__":
    main()
