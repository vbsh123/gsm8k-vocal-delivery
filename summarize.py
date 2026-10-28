#!/usr/bin/env python3
"""Print per-style accuracy from evaluator JSONL results; no GPU required."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def print_summary(path, *additional_paths):
    # Evaluation appends on reruns; count the latest result for each clip once.
    latest = {}
    for results_path in (path, *additional_paths):
        with Path(results_path).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    latest[row["clip_id"]] = row

    groups = defaultdict(list)
    for row in latest.values():
        groups[row["style"]].append(row)

    print("\nAccuracy by style (latest result per clip)")
    print(f"{'Style':<14} {'Correct / total':>16} {'Accuracy':>10} {'Parse failures':>15}")
    for style, rows in sorted(groups.items()):
        correct = sum(row["correct"] is True for row in rows)
        failures = sum(row.get("parse_failed", False) is True for row in rows)
        count = len(rows)
        print(f"{style:<14} {f'{correct} / {count}':>16} {100 * correct / count:>9.1f}% {failures:>15}")
    if not groups:
        print("No evaluated clips.")
    print("Parse failures count as incorrect. Skipped questions are excluded.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="Results JSONL files to summarize together")
    print_summary(*parser.parse_args().results)


if __name__ == "__main__":
    main()
