"""Reconstruct question IDs used by the both-pass vector extractor."""

import argparse
from collections import Counter
import json
from pathlib import Path


def extraction_questions(records, labels):
    recordings = {r["clip_id"]: r for r in records}
    classifications = {r["clip_id"]: r["predicted_style"] for r in labels}
    neutrals = {(r["question_id"], r["speaker"], r["repeat"]): r
                for r in recordings.values() if r["style"] == "neutral"}
    questions = set()
    pairs = Counter()
    # Match extract_vectors.py: emotional style must be NO, neutral must be YES.
    for result in labels:
        if result["style"] not in ["angry", "happy", "confident"] or result["predicted_style"] != "no":
            continue
        r = recordings[result["clip_id"]]
        neutral = neutrals[r["question_id"], r["speaker"], r["repeat"]]
        if classifications.get(neutral["clip_id"]) == "yes":
            questions.add(r["question_id"])
            pairs[r["style"]] += 1
    return questions, dict(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/manual200-mixed/cropped/manifest.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("outputs/manual200-mixed/extraction-labels.jsonl"),
                        help="Unchanged classification snapshot from vector extraction, not newly extended labels")
    parser.add_argument("--output", type=Path, default=Path("outputs/manual200-mixed/vector-extraction-questions.jsonl"))
    args = parser.parse_args()
    def read(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    questions, pairs = extraction_questions(read(args.manifest), read(args.labels))
    if not questions:
        parser.error("No both-pass extraction pairs found; check the classification snapshot")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps({"question_id": q}) + "\n" for q in sorted(questions)))
    print(f"Extraction pairs: {pairs}")
    print(f"Excluded {len(questions)} distinct extraction questions: {args.output}")


if __name__ == "__main__":
    main()
