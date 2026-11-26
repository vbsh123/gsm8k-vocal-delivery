"""Generate Jenny terrified speech for the two archived math cohorts, then score every clip."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave

SIZES = {"gsm8k": 1319, "multiarith": 580}
OUTPUT = Path("outputs/jenny-terrified-math")
PILOT = Path("outputs/jenny-terrified-gsm30")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reuse_pilot(folder):
    if not (PILOT / "plan.jsonl").exists():
        return
    cached = {(r["question_id"], r["request"]): PILOT / r["audio_path"]
              for r in read_rows(PILOT / "plan.jsonl")}
    reused = 0
    for row in read_rows(folder / "plan.jsonl"):
        source = cached.get((row["question_id"], row["request"]))
        target = folder / row["audio_path"]
        if not source or not source.is_file() or target.exists():
            continue
        try:
            with wave.open(str(source)) as audio:
                frames = audio.getnframes()
                if not frames or len(audio.readframes(frames)) != frames * audio.getnchannels() * audio.getsampwidth():
                    continue
        except (EOFError, wave.Error):
            continue
        temporary = target.with_suffix(".wav.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        reused += 1
    print(f"GSM8K: reused {reused} pilot recordings with identical synthesis requests.", flush=True)


def prepare(root):
    banks = json.loads(Path("data/jenny_math_question_supplement.json").read_text())
    banks["gsm8k"].extend(json.loads(Path("kimi_gsm8k_questions.json").read_text())["questions"])
    commands = []
    for dataset, expected in SIZES.items():
        rows = banks[dataset]
        ids = {r["question_id"] for r in rows}
        if len(rows) != expected or len(ids) != expected:
            raise ValueError(f"{dataset}: expected {expected} unique questions")
        folder = root / dataset
        folder.mkdir(parents=True, exist_ok=True)
        questions = folder / "questions.jsonl"
        questions.write_text("".join(json.dumps(row) + "\n" for row in rows))
        cmd = [sys.executable, "-u", "cloud_tts_smoke.py", "--provider", "azure",
               "--input-jsonl", str(questions), "--num-examples", str(expected),
               "--azure-voice", "en-US-JennyNeural", "--azure-intensity", "1.5",
               "--styles", "terrified", "--output", str(folder), "--retry-passes", "5", "--resume"]
        subprocess.run([*cmd, "--prepare-only"], check=True)
        commands.append(cmd)
    reuse_pilot(root / "gsm8k")
    return commands


def summarize(root):
    complete = True
    for dataset, expected in SIZES.items():
        folder = root / dataset
        plan = folder / "plan.jsonl"
        scores = folder / "results-emotion2vec.jsonl"
        tasks = read_rows(plan) if plan.exists() else []
        by_path = {row["audio_path"]: row["question_id"] for row in tasks}
        labels = {}
        for row in read_rows(scores) if scores.exists() else []:
            path = row["audio_path"]
            if path not in by_path or row["question_id"] != by_path[path] or row["requested_style"] != "terrified":
                raise ValueError(f"{scores}: unmatched prediction for {path}")
            labels[path] = row["emotion"]
        counts = Counter(labels.values())
        clips = sum((folder / path).is_file() for path in by_path)
        n = len(labels)
        print(f"{dataset}: {clips}/{expected} WAVs, {n}/{expected} E2V predictions.")
        if n:
            print(f"  fearful: {counts['fearful']}/{n} ({counts['fearful']/n:.1%}); labels: {dict(counts.most_common())}")
        complete &= clips == n == expected
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "summary"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.action == "summary":
        summarize(args.output)
        return
    commands = prepare(args.output)
    if args.action == "prepare":
        return
    classifier = os.environ.get("EMOTION_PYTHON", ".venv-emotion/bin/python")
    if not shutil.which(classifier):
        parser.error(f"Missing E2V interpreter {classifier}. Set EMOTION_PYTHON to the pilot's Python.")
    for command in commands:
        subprocess.run(command, check=True)
    for dataset in SIZES:
        print(f"Scoring {dataset} with Emotion2Vec on CPU.", flush=True)
        subprocess.run([classifier, "-u", "classify_cloud_smoke.py", "--device", "cpu",
                        "--path", str(args.output / dataset)], check=True)
    if not summarize(args.output):
        sys.exit("Incomplete audio or E2V coverage. Run again to resume.")


if __name__ == "__main__":
    main()
