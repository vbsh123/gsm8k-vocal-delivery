"""Compare saved baseline/steered answers grouped by input emotion labels. No inference."""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import re


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def flexible_correct(row):
    # Same last-number extraction and numeric comparison used in our prior analyses.
    matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", row["response"])
    if not matches:
        return False
    value = next(x for x in matches[-1] if x).replace(",", "").replace("$", "")
    try:
        return Fraction(re.sub(r"\.$", "", value)) == Fraction(
            str(row["reference_answer"]).replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return False


def accuracy(count, n):
    value = count / n
    margin = 1.96 * math.sqrt(value * (1 - value) / n)
    return f"{count}/{n} = {value:.1%} [95% CI {max(0, value-margin):.1%}, {min(1, value+margin):.1%}]"


def paired_stats(ids, baseline, steered, score):
    pairs = [(score(steered[k]), score(baseline[k])) for k in ids]
    n = len(pairs)
    if not n:
        return None
    wins = sum(s and not b for s, b in pairs)
    losses = sum(b and not s for s, b in pairs)
    d = wins + losses
    p = min(1, 2 * sum(math.comb(d, k) for k in range(min(wins, losses) + 1)) / 2**d)
    return {"n": n, "steered": sum(s for s, b in pairs),
            "baseline": sum(b for s, b in pairs), "wins": wins, "losses": losses,
            "gain_points": 100 * (wins-losses) / n, "mcnemar_p": p}


def build_groups(ids, diffa, emotion2vec):
    # Missing predictions never become "not neutral". All selections use original
    # audio labels, not generated answers or their correctness.
    return {
        "All paired questions": list(ids),
        "DIFFA: neutral": [k for k in ids if diffa.get(k, {}).get("predicted_style") == "yes"],
        "DIFFA: not neutral": [k for k in ids if diffa.get(k, {}).get("predicted_style") == "no"],
        "DIFFA: missing/invalid": [k for k in ids
                                   if diffa.get(k, {}).get("predicted_style") not in ("yes", "no")],
        "emotion2vec: neutral": [k for k in ids
                                 if emotion2vec.get(k, {}).get("binary_prediction") == "neutral"],
        "emotion2vec: not neutral": [k for k in ids
                                     if emotion2vec.get(k, {}).get("binary_prediction") == "emotional"],
        "emotion2vec: unresolved": [k for k in ids
                                    if emotion2vec.get(k, {}).get("binary_prediction") == "unresolved"],
        "emotion2vec: missing/invalid": [k for k in ids if emotion2vec.get(k, {}).get("binary_prediction")
                                        not in ("neutral", "emotional", "unresolved")],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path("outputs/steer-angry-L0-15-dev200-a1/results.jsonl"))
    parser.add_argument("--path", type=Path, default=Path("outputs/manual200-mixed/cropped"))
    parser.add_argument("--condition", default="L4-acoustic-a1")
    parser.add_argument("--scoring", choices=("strict", "non-strict", "both"), default="both")
    args = parser.parse_args()
    rows = read_rows(args.results)
    baseline = {r["clip_id"]: r for r in rows if r["condition"] == "baseline"}
    steered = {r["clip_id"]: r for r in rows if r["condition"] == args.condition}
    if not baseline or not steered:
        parser.error("Results must contain baseline and the selected steering condition")
    ids = sorted(baseline.keys() & steered.keys())
    diffa = {r["clip_id"]: r for r in read_rows(args.path / "results-neutral-binary.jsonl")}
    emotion2vec = {r["clip_id"]: r for r in read_rows(args.path / "results-emotion2vec-neutral.jsonl")}
    print(f"Condition: {args.condition}; {len(ids)} paired questions.")
    print(f"Without a partner: {len(baseline.keys()-steered.keys())} baseline rows, "
          f"{len(steered.keys()-baseline.keys())} steered rows.")
    scores = {"strict": lambda r: r["correct"], "non-strict": flexible_correct}
    for group, selected in build_groups(ids, diffa, emotion2vec).items():
        print(f"\n{group}: {len(selected)} pairs")
        for mode, score in scores.items():
            if args.scoring != "both" and args.scoring != mode:
                continue
            stats = paired_stats(selected, baseline, steered, score)
            if stats is None:
                continue
            print(f"  {mode}: Steered {accuracy(stats['steered'], stats['n'])}")
            print(f"         Baseline {accuracy(stats['baseline'], stats['n'])}")
            print(f"         Gain {stats['gain_points']:+.1f} points; "
                  f"wins/losses {stats['wins']}/{stats['losses']}; "
                  f"exact McNemar p={stats['mcnemar_p']:.4f}")
    print("\nLabels describe the original input audio, not the steered output.")
    print("Classifier groups may overlap; subgroup p-values are exploratory and unadjusted.")


if __name__ == "__main__":
    main()
