"""Classify cloud TTS samples with the existing emotion2vec or SenseVoice setup."""

import argparse
from collections import Counter
import json
import importlib
from pathlib import Path

def summarize(rows, model="emotion2vec"):
    print(f"\n{model} classification summary")
    labels = sorted({row["emotion"] for row in rows})
    print(f'{"Requested":12} {"Clips":>6}' + ''.join(f'{label:>12}' for label in labels) + f'{"Match":>10}')
    for style in dict.fromkeys(row["requested_style"] for row in rows):
        counts = Counter(row["emotion"] for row in rows if row["requested_style"] == style)
        n = sum(counts.values())
        match = f'{counts[style]/n:.1%}' if style in {"neutral", "angry", "happy", "sad"} else "n/a"
        print(f'{style:12} {n:6}' + ''.join(f'{counts[label]:12}' for label in labels) + f'{match:>10}')
    print("Cells are clip counts; match means predicted emotion equals requested style.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("outputs/azure-guy-smoke"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", choices=["emotion2vec", "sensevoice"], default="emotion2vec")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    output = args.path / f"results-{args.model}.jsonl"
    if args.summary_only:
        summarize([json.loads(line) for line in output.read_text().splitlines() if line.strip()], args.model)
        return
    tasks = [json.loads(line) for line in (args.path / "plan.jsonl").read_text().splitlines()
             if line.strip()]
    import librosa
    import soundfile as sf

    adapter = importlib.import_module(f"{args.model}_agreement")
    saved = {r['audio_path']:r for r in (json.loads(line) for line in output.read_text().splitlines() if line.strip())} if output.exists() else {}
    if any(r['model'] != adapter.MODEL for r in saved.values()):
        parser.error('Saved predictions use another model')
    pending = [t for t in tasks if t['audio_path'] not in saved]
    rows = [saved[t['audio_path']] for t in tasks if t['audio_path'] in saved]
    print(f'{len(tasks)} clips; {len(pending)} classification predictions pending.', flush=True)
    predict = adapter.load_predictor(args.device) if pending else None
    with output.open("a") as stream:
        print(f'{"Provider":12} {"Requested":12} {"Predicted":12} {"Top score":>10}', flush=True)
        for task in pending:
            audio, sample_rate = sf.read(args.path / task["audio_path"], dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            prediction = predict(audio)
            row = {"provider": task["provider"], "requested_style": task["style"],
                   "voice": task["voice"], "audio_path": task["audio_path"],
                   "question_id": task.get("question_id"),
                   "model": adapter.MODEL, **prediction}
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            score = f'{max(prediction["scores"]):.1%}' if "scores" in prediction else "n/a"
            print(f'{task["provider"]:12} {task["style"]:12} {prediction["emotion"]:12} '
                  f'{score:>10}', flush=True)
    summarize(rows, args.model)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
