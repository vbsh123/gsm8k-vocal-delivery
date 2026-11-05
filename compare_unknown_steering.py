"""Compare question-level fixes and regressions of the leading unknown-steering conditions."""

import argparse
from itertools import combinations
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path(
        "outputs/steer-unknown-emotion2vec-neutral-L0-25/results.jsonl"))
    p.add_argument("--conditions", nargs="+", default=[
        "L9-acoustic-a-1", "L5-answer-a-1", "L6-answer-a-1"])
    args = p.parse_args(argv)
    rows = {}
    with args.results.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A live run may still be writing its last row.
            rows[row["clip_id"], row["condition"]] = row
    ids = sorted({clip for clip, condition in rows
                  if condition == "baseline" and
                  all((clip, c) in rows for c in args.conditions)})
    if not ids:
        p.error("No questions have baseline and all requested conditions")
    base = sum(rows[clip, "baseline"]["correct"] for clip in ids)
    print(f"Common paired questions: {len(ids)}; baseline {base}/{len(ids)} = {base/len(ids):.1%}")
    changes = {}
    for condition in args.conditions:
        wins = {clip for clip in ids if rows[clip, condition]["correct"]
                and not rows[clip, "baseline"]["correct"]}
        losses = {clip for clip in ids if not rows[clip, condition]["correct"]
                  and rows[clip, "baseline"]["correct"]}
        changes[condition] = {"wins": wins, "losses": losses}
        correct = sum(rows[clip, condition]["correct"] for clip in ids)
        print(f"\n{condition}: {correct}/{len(ids)} = {correct/len(ids):.1%}; "
              f"wins/losses={len(wins)}/{len(losses)}")
        for label, clips in (("FIXED", wins), ("BROKE", losses)):
            print(f"  {label}:")
            for clip in sorted(clips):
                b, s = rows[clip, "baseline"], rows[clip, condition]
                print(f"    {b['question_id']}: baseline={b.get('predicted_answer')} "
                      f"steered={s.get('predicted_answer')}")
            if not clips:
                print("    none")
    for kind in ("wins", "losses"):
        print(f"\nShared {kind}:")
        for a, b in combinations(args.conditions, 2):
            shared = changes[a][kind] & changes[b][kind]
            names = [rows[clip, "baseline"]["question_id"] for clip in sorted(shared)]
            print(f"  {a} & {b}: {len(shared)} — {', '.join(names) or 'none'}")
        shared = set.intersection(*(changes[c][kind] for c in args.conditions))
        names = [rows[clip, "baseline"]["question_id"] for clip in sorted(shared)]
        print(f"  ALL: {len(shared)} — {', '.join(names) or 'none'}")
    print("\nQuestions where the selected conditions disagree on correctness:")
    different = 0
    for clip in ids:
        if len({rows[clip, c]["correct"] for c in args.conditions}) > 1:
            different += 1
            outcomes = ", ".join(f"{c}={'correct' if rows[clip,c]['correct'] else 'wrong'}"
                                 for c in args.conditions)
            print(f"  {rows[clip, 'baseline']['question_id']}: {outcomes}")
    if not different:
        print("  none — these conditions have identical correctness outcomes")


if __name__ == "__main__":
    main()
