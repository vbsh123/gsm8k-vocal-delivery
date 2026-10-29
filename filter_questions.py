"""Keep questions whose previous recordings all fit the duration limit."""

import argparse
import json
from pathlib import Path

import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    questions = []
    for source in args.sources:
        recordings = [json.loads(line) for line in (source.parent / "manifest.jsonl").read_text().splitlines() if line.strip()]
        long_questions = set()
        for recording in recordings:
            info = sf.info(source.parent / recording["audio_path"])
            if info.frames > args.max_seconds * info.samplerate:
                long_questions.add(recording["question_id"])
        for line in source.read_text().splitlines():
            question = json.loads(line)
            if question["question_id"] not in long_questions:
                questions.append(question)

    selected = questions[:args.num_examples]
    args.output.write_text("".join(json.dumps(q) + "\n" for q in selected))
    print(len(selected))


if __name__ == "__main__":
    main()
