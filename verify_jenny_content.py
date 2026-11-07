"""Audit spoken question content using transcripts already saved by SenseVoice."""

import argparse
from collections import defaultdict
import csv
from difflib import SequenceMatcher
import json
from pathlib import Path
import re


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def clean_transcript(text):
    return re.sub(r"<\|[^|]*\|>", " ", text).strip()


def normalize(text, normalizer):
    # Whisper's normalizer normally drops bracketed content and numeric minus
    # signs. Preserve the words in parentheses and explicitly spoken negatives.
    text = clean_transcript(text).replace("−", "-")
    text = re.sub(r"(?<!\w)-(?=\d)", " minus ", text)
    text = re.sub(r"[()\[\]]", " ", text)
    text = text.replace("<", " less than ").replace(">", " greater than ")
    return " ".join(normalizer(text).split())


def numbers(text):
    # Whisper writes the value 1 as 'one'. Keep order/repetitions, not just sets.
    return ["1" if token == "one" else token
            for token in re.findall(r"[+-]?\d+(?:\.\d+)?|\bone\b", text)]


def word_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1,
                               previous[j-1] + (expected != actual)))
        previous = current
    return previous[-1]


def compare(reference, transcript, normalizer):
    expected = normalize(reference, normalizer)
    recognized = normalize(transcript, normalizer)
    a, b = expected.split(), recognized.split()
    edits = []
    for tag, i, j, k, l in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            edits.append({"type": tag, "expected": " ".join(a[i:j]), "heard": " ".join(b[k:l])})
    expected_numbers, recognized_numbers = numbers(expected), numbers(recognized)
    distance = word_distance(a, b)
    return {"normalized_reference": expected, "normalized_transcript": recognized,
            "normalized_exact_match": expected == recognized,
            "word_errors": distance, "reference_words": len(a), "wer": distance / max(1, len(a)),
            "reference_numbers": expected_numbers, "transcript_numbers": recognized_numbers,
            "number_mismatch": expected_numbers != recognized_numbers, "edits": edits}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["style"]].append(row)
    print("\nContent audit from saved SenseVoice transcripts")
    print(f'{"Delivery":12} {"Selected":>9} {"Checked":>8} {"Missing":>8} '
          f'{"Text exact":>12} {"Numbers same":>13} {"Corpus WER":>12}')
    for style, selected in sorted(groups.items()):
        checked = [r for r in selected if r["status"] == "checked"]
        n = len(checked)
        exact = f'{sum(r["normalized_exact_match"] for r in checked)/n:.1%}' if n else "n/a"
        numeric = f'{sum(not r["number_mismatch"] for r in checked)/n:.1%}' if n else "n/a"
        words = sum(r["reference_words"] for r in checked)
        wer = f'{sum(r["word_errors"] for r in checked)/words:.1%}' if words else "n/a"
        print(f'{style:12} {len(selected):9} {n:8} {len(selected)-n:8} '
              f'{exact:>12} {numeric:>13} {wer:>12}')
    print("Text exact ignores normalized spelling/number-format differences; WER is word error rate.")
    print("ASR disagreement flags clips for listening; it does not prove a TTS error.")
    print("Number matching is a normalized token check, not a semantic check of fractions, units, or meaning.")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", type=Path, nargs="+", default=[Path("outputs/azure-jenny500-extra"),
        Path("outputs/azure-jenny500-neutral"), Path("outputs/azure-jenny-next500")])
    p.add_argument("--output", type=Path, default=Path("outputs/jenny-content-check"))
    p.add_argument("--evaluated-only", action="store_true", help="Check only clips with a saved GSM8K evaluation.")
    p.add_argument("--styles", nargs="+", help="Default: all styles in the plans.")
    p.add_argument("--show", type=int, default=5, help="Number of suspect clips to print.")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args(argv)
    if args.summary_only:
        summarize(read(args.output / "results.jsonl"))
        return

    # Text normalization only: no model, tokenizer download, or GPU allocation.
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    normalizer = EnglishTextNormalizer({})
    rows = []
    for folder in args.paths:
        predictions_path = folder / "results-sensevoice.jsonl"
        predictions = {r["audio_path"]: r for r in read(predictions_path)} if predictions_path.exists() else {}
        evaluated = None
        if args.evaluated_only:
            evaluated = {r["audio_path"] for r in read(folder / "results-matched-accuracy.jsonl")}
        for task in read(folder / "plan.jsonl"):
            if args.styles and task["style"] not in args.styles:
                continue
            if evaluated is not None and task["audio_path"] not in evaluated:
                continue
            transcript = predictions.get(task["audio_path"], {}).get("raw_text")
            row = {"source": str(folder), "audio_path": task["audio_path"],
                   "audio_file": str((folder / task["audio_path"]).resolve()),
                   "question_id": task["question_id"], "voice": task["voice"], "style": task["style"],
                   "reference": task["question"], "transcript": clean_transcript(transcript or ""),
                   "asr_model": "FunAudioLLM/SenseVoiceSmall",
                   "status": "checked" if transcript is not None else "missing_transcript"}
            if transcript is not None:
                row.update(compare(task["question"], transcript, normalizer))
            rows.append(row)
        print(f"Checked saved transcripts from {folder}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    suspects = sorted((r for r in rows if r["status"] == "checked" and not r["normalized_exact_match"]),
                      key=lambda r: (not r["number_mismatch"], -r["wer"]))
    with (args.output / "review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["question_id", "style", "number_mismatch", "wer", "audio_file", "reference",
                         "transcript", "reference_numbers", "transcript_numbers", "edits"])
        for row in suspects:
            writer.writerow([row["question_id"], row["style"], row["number_mismatch"], row["wer"],
                             row["audio_file"], row["reference"], row["transcript"],
                             json.dumps(row["reference_numbers"]), json.dumps(row["transcript_numbers"]),
                             json.dumps(row["edits"], ensure_ascii=False)])
    summarize(rows)
    for row in suspects[:max(0, args.show)]:
        print(f'\n{row["question_id"]} / {row["style"]} / WER={row["wer"]:.1%}')
        print(f'  Numbers: expected {row["reference_numbers"]}; transcript {row["transcript_numbers"]}')
        for edit in row["edits"][:6]:
            print(f'  {edit["type"]}: {edit["expected"]!r} -> {edit["heard"]!r}')
        print(f'  Audio: {row["audio_file"]}')
    print(f'\nSaved all comparisons: {args.output / "results.jsonl"}')
    print(f'Saved {len(suspects)} text-mismatch candidates: {args.output / "review.csv"}')


if __name__ == "__main__":
    main()
