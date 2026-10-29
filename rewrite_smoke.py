"""Add handwritten style remarks to questions and save text for a separate TTS run."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=ROOT / "examples/tiny.jsonl")
    parser.add_argument("--style", default="angry")
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--remark-index", type=int, default=0, choices=[0, 1])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/wording-smoke")
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.input_jsonl.read_text().splitlines() if line.strip()]
    remarks = json.loads((ROOT / "configs/remarks.json").read_text())[args.style]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "tts-input.jsonl").open("w") as tts, (args.output / "rewrites.jsonl").open("w") as saved:
        for index, question in enumerate(questions[:args.num_examples]):
            template = remarks[(args.remark_index + index) % len(remarks)]
            original = question.get("question", question.get("text"))
            text = template.format(question=original)
            tts.write(json.dumps({
                "id": question.get("id", question.get("question_id", f"local_{index:05d}")),
                "question": text, "answer": question["answer"],
            }) + "\n")
            saved.write(json.dumps({
                "original": original, "style": args.style, "template": template, "text": text,
            }) + "\n")
            print(text + "\n", flush=True)


if __name__ == "__main__":
    main()
