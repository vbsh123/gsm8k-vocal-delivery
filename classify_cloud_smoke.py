"""Classify cloud TTS smoke samples with the existing emotion2vec+ Large setup."""

import argparse
import json
from pathlib import Path

from emotion2vec_agreement import MODEL, load_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("outputs/azure-guy-smoke"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    tasks = [json.loads(line) for line in (args.path / "plan.jsonl").read_text().splitlines()
             if line.strip()]
    import librosa
    import soundfile as sf

    predict = load_predictor(args.device)
    output = args.path / "results-emotion2vec.jsonl"
    with output.open("w") as stream:
        print(f'{"Provider":12} {"Requested":12} {"Predicted":12} {"Top score":>10}', flush=True)
        for task in tasks:
            audio, sample_rate = sf.read(args.path / task["audio_path"], dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            prediction = predict(audio)
            row = {"provider": task["provider"], "requested_style": task["style"],
                   "voice": task["voice"], "audio_path": task["audio_path"],
                   "model": MODEL, **prediction}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f'{task["provider"]:12} {task["style"]:12} {prediction["emotion"]:12} '
                  f'{max(prediction["scores"]):10.1%}', flush=True)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
