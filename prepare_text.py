"""Add handwritten style remarks to questions and save text for a separate TTS run."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=ROOT / "examples/tiny.jsonl")
    parser.add_argument("--style", default="angry", help="Style name, or all for all seven styles")
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--remark-index", type=int, default=0, choices=[0, 1])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/manual-angry")
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.input_jsonl.read_text().splitlines() if line.strip()]
    remarks = json.loads((ROOT / "configs/remarks.json").read_text())
    selected = remarks if args.style == "all" else {args.style: remarks[args.style]}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "tts-input.jsonl").open("w") as tts:
        for index, question in enumerate(questions[:args.num_examples]):
            original = question.get("question", question.get("text"))
            style_texts = {
                style: templates[(args.remark_index + index) % len(templates)].format(question=original)
                for style, templates in selected.items()
            }
            tts.write(json.dumps({
                "id": question.get("id", question.get("question_id", f"local_{index:05d}")),
                "question": original, "answer": question["answer"], "style_texts": style_texts,
            }) + "\n")
            for style, text in style_texts.items():
                print(f"{style}: {text}\n", flush=True)


if __name__ == "__main__":
    main()
