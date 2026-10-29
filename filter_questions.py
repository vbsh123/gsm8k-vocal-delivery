"""Keep previously short questions and optionally add new GSM8K questions."""

import argparse
import json
from pathlib import Path
import random
import sys

import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--add-new", action="store_true", help="Fill the requested count with unseen GSM8K test questions")
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    questions = []
    excluded = set()
    for source in args.sources:
        recordings = [json.loads(line) for line in (source.parent / "manifest.jsonl").read_text().splitlines() if line.strip()]
        long_questions = set()
        for recording in recordings:
            info = sf.info(source.parent / recording["audio_path"])
            if info.frames > args.max_seconds * info.samplerate:
                long_questions.add(recording["question_id"])
        for line in source.read_text().splitlines():
            question = json.loads(line)
            excluded.add(question["question_id"])
            if question["question_id"] not in long_questions:
                questions.append(question)

    selected = questions[:args.num_examples]
    retained = len(selected)
    if args.add_new and retained < args.num_examples:
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main", split="test")
        new = [
            {"question_id": f"gsm8k_test_{i:05d}", "text": row["question"],
             "answer": row["answer"].rsplit("####", 1)[-1].strip(), "source_answer": row["answer"]}
            for i, row in enumerate(dataset)
            if f"gsm8k_test_{i:05d}" not in excluded
        ]
        random.Random(args.seed).shuffle(new)
        selected.extend(new[:args.num_examples - retained])
    args.output.write_text("".join(json.dumps(q) + "\n" for q in selected))
    print(f"Retained {retained} previous questions; added {len(selected) - retained} new questions.", file=sys.stderr)
    print(len(selected))


if __name__ == "__main__":
    main()
