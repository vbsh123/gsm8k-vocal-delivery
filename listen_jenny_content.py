"""Copy a few flagged Jenny clips and neutral counterparts into a portable listening folder."""

import argparse
from html import escape
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("outputs/jenny-content-check/results.jsonl"))
    p.add_argument("--output", type=Path, default=Path("outputs/jenny-content-listen"))
    p.add_argument("--limit", type=int, default=5, help="Worst text-mismatch clips, before adding neutral counterparts.")
    p.add_argument("--questions", nargs="+", help="Optionally restrict selection to these question IDs.")
    args = p.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    suspects = [r for r in rows if r["status"] == "checked" and not r["normalized_exact_match"]
                and (not args.questions or r["question_id"] in args.questions)]
    suspects.sort(key=lambda r: (not r["number_mismatch"], -r["wer"]))
    neutral = {(r["question_id"], r["voice"]): r for r in rows if r["style"] == "neutral"}
    args.output.mkdir(parents=True, exist_ok=True)
    copied, cards = {}, []

    def player(row):
        source = Path(row["audio_file"])
        if str(source) not in copied:
            name = f"clip_{len(copied):03d}{source.suffix}"
            shutil.copy2(source, args.output / name)
            copied[str(source)] = {**row, "listening_file": name}
        name = copied[str(source)]["listening_file"]
        transcript = escape(row.get("transcript", ""))
        return (f'<h3>{escape(row["style"])}</h3><audio controls preload="none" src="{name}"></audio>'
                f'<p><b>SenseVoice:</b> {transcript}</p>')

    for row in suspects[:max(0, args.limit)]:
        partner = neutral.get((row["question_id"], row["voice"]))
        card = (f'<section><h2>{escape(row["question_id"])} / {escape(row["style"])}</h2>'
                f'<p><b>Expected:</b> {escape(row["reference"])}</p>'
                f'<p>Normalized WER: {row["wer"]:.1%}; number mismatch: {row["number_mismatch"]}</p>')
        if partner and partner["audio_file"] != row["audio_file"]:
            card += player(partner)
        elif row["style"] != "neutral":
            card += '<p>Neutral counterpart was absent from the audited subset.</p>'
        card += player(row)
        cards.append(card + '</section>')
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Jenny content listening review</title>'
        '<style>body{font:17px system-ui;max-width:1000px;margin:32px auto;padding:0 20px}'
        'section{border-top:1px solid #bbb;margin-top:32px;padding-top:12px}audio{width:100%}</style>'
        '<h1>Jenny content listening review</h1>'
        '<p>These are selected ASR mismatches, not a representative sample. Listen for the actual words '
        'and quantities; SenseVoice can be wrong. Neutral counterparts are included where available.</p>'
        + '\n'.join(cards), encoding="utf-8")
    (args.output / "manifest.jsonl").write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in copied.values()), encoding="utf-8")
    print(f"Copied {len(copied)} audio files; originals unchanged.")
    print(f"Download this whole folder: {args.output}")
    print("Then open index.html locally to listen and compare the text.")


if __name__ == "__main__":
    main()
