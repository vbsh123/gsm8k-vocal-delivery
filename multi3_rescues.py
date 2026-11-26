"""Compare original angry and rewritten neutral on the same original-neutral failures."""
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
METRIC = "flexible-numeric"
STYLES = ("neutral", "angry")
TYPES = ("Ready only", "Other, no extracted number", "Wrong extracted number")


def load(folder, ids):
    rows = {}
    for line in (folder / "rescored.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = r["question_id"], r["condition"]
        if key[0] not in ids or key[1] not in STYLES:
            continue
        if key in rows or r["metrics"][METRIC] not in (False, True):
            raise ValueError(f"Invalid or duplicate scored row {key} in {folder}")
        rows[key] = r
    if len(rows) != len(ids) * len(STYLES):
        raise ValueError(f"{folder}: found {len(rows)} matched rows, expected {len(ids)*2}.")
    return rows


def failure(response):
    if re.fullmatch(r"\s*Ready[.!?]?\s*", response["response"], re.I):
        return TYPES[0]
    if response["predictions"]["flexible-extract"] == "[invalid]":
        return TYPES[1]
    return TYPES[2]


def compare(old, new, ids):
    groups = {name: Counter() for name in TYPES}
    losses = 0
    for q in ids:
        for style in STYLES:
            a, b = old[q, style], new[q, style]
            if any(a[k] != b[k] for k in ("audio_sha256", "reference_answer")) or a["prompt"] == b["prompt"]:
                raise ValueError(f"{q}/{style}: expected identical recording and gold under different prompts.")
        n, a, rewritten = old[q, "neutral"], old[q, "angry"], new[q, "neutral"]
        if n["metrics"][METRIC]:
            losses += int(not rewritten["metrics"][METRIC])
        else:
            groups[failure(n)][bool(a["metrics"][METRIC]), bool(rewritten["metrics"][METRIC])] += 1
    total = sum(groups.values(), Counter())
    print(f"{len(ids)} matched questions, flexible-numeric. Among original-neutral misses:")
    print(f"{'Original neutral response':29} {'Both fix':>9} {'Angry only':>11} {'Rewrite only':>13} {'Neither':>9}")
    for name, c in [*groups.items(), ("Total", total)]:
        print(f"{name:29} {c[True,True]:9} {c[True,False]:11} {c[False,True]:13} {c[False,False]:9}")
    fixed = total[True,True] + total[False,True]
    print(f"Original angry also correct on {total[True,True]}/{fixed} neutral rewrite fixes.")
    print(f"Neutral rewrite: {fixed} wrong-to-right, {losses} right-to-wrong, net {fixed-losses:+d}.")


def main():
    ids = list(json.loads((ROOT / "data/multiarith_original_jenny290.json").read_text())["questions"])[20:]
    if len(ids) != 270:
        raise ValueError("Expected the 270 questions following the pilot.")
    folders = [ROOT / "outputs" / f"diffa-multiarith-{suffix}" for suffix in
               ("three-asks290", "sequenced-remaining270")]
    for folder in folders:
        if not (folder / "rescored.jsonl").is_file():
            raise SystemExit(f"Missing {folder}/rescored.jsonl. Run the corresponding MultiArith summary first.")
    old, new = [load(folder, set(ids)) for folder in folders]
    compare(old, new, ids)


if __name__ == "__main__":
    main()
