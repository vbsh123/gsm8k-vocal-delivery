"""Stage Jenny terrified IFEval audio and export it with the existing terrified math recordings."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import wave
import zipfile

from jenny_terrified_math import OUTPUT as MATH, SIZES, read_rows
from run_spoken_ifeval import REVISION, setup_official

ROOT = Path("outputs/jenny-terrified-ifeval")
ARCHIVE = Path("outputs/jenny-terrified-math-ifeval.zip")
BANK_SHA = "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49"
VOICE = "en-US-JennyNeural"
STYLE = "terrified"
RATES = (-10, 0, 10)
CARRIER = "Here is the task."


def folders():
    return {**{MATH / name: count for name, count in SIZES.items()},
            ROOT / "spoken": 541,
            **{ROOT / "flipped" / f"rate{rate:+d}": 1 for rate in RATES}}


def synthesize(prepare_only):
    bank = setup_official() / "data/input_data.jsonl"
    if hashlib.sha256(bank.read_bytes()).hexdigest() != BANK_SHA:
        raise ValueError(f"Unexpected official IFEval question bank at {bank} (revision {REVISION})")
    questions = [dict(question_id=f'ifeval_{e["key"]}', question=e["prompt"], **e) for e in read_rows(bank)]
    if len(questions) != 541 or len({r["question_id"] for r in questions}) != 541:
        raise ValueError("Expected 541 distinct IFEval prompts")
    carrier = [dict(question_id="carrier", question=CARRIER, answer="")]
    jobs = [(ROOT / "spoken", questions, 0)]
    jobs.extend((ROOT / "flipped" / f"rate{r:+d}", carrier, r) for r in RATES)
    for folder, rows, rate in jobs:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "questions.jsonl"
        if path.exists() and read_rows(path) != rows:
            raise ValueError(f"Saved question bank differs at {path}")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        command = [sys.executable, "-u", "cloud_tts_smoke.py", "--provider", "azure", "--styles", STYLE,
                   "--azure-voice", VOICE, "--azure-intensity", "1.5", "--azure-rate-offset", str(rate),
                   "--input-jsonl", str(path), "--num-examples", str(len(rows)), "--output", str(folder),
                   "--resume", "--retry-passes", "5"]
        if prepare_only:
            command.append("--prepare-only")
        subprocess.run(command, check=True)


def archive():
    paths = []
    for folder, count in folders().items():
        rows = read_rows(folder / "plan.jsonl")
        names = {r["audio_path"] for r in rows}
        if (len(rows) != count or len(names) != count or len({r["question_id"] for r in rows}) != count
                or any(Path(n).name != n or not n.endswith(".wav") for n in names)
                or any(r["style"] != STYLE or r["voice"] != VOICE or r["provider"] != "azure" for r in rows)):
            raise ValueError(f"Unexpected terrified audio plan at {folder}")
        for name in sorted(names):
            path = folder / name
            with wave.open(str(path), "rb") as wav:
                frames = wav.getnframes()
                if not frames or len(wav.readframes(frames)) != frames * wav.getsampwidth() * wav.getnchannels():
                    raise ValueError(f"Incomplete WAV at {path}")
            paths.append(path)
        paths.extend([folder / "questions.jsonl", folder / "plan.jsonl"])
        for name in ("listen.html", "results-emotion2vec.jsonl"):
            if (folder / name).is_file():
                paths.append(folder / name)
    with TemporaryDirectory(dir=ARCHIVE.parent) as tmp:
        staged = Path(tmp) / ARCHIVE.name
        with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for path in paths:
                z.write(path, path.as_posix())
        with zipfile.ZipFile(staged) as z:
            bad = z.testzip()
            if bad:
                raise ValueError(f"Archive CRC failed for {bad}")
        checksum = hashlib.sha256()
        with staged.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        sidecar = Path(tmp) / (ARCHIVE.name + ".sha256")
        sidecar.write_text(f"{checksum.hexdigest()}  {ARCHIVE.name}\n")
        staged.replace(ARCHIVE)
        sidecar.replace(ARCHIVE.with_suffix(".zip.sha256"))
    print(f"Verified {sum(folders().values())} WAVs. Download {ARCHIVE} and {ARCHIVE}.sha256", flush=True)


@contextmanager
def lock():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "worker.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Terrified IFEval worker already active.")
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "prepare", "summary", "pack", "e2v"])
    args = parser.parse_args()
    if args.action == "summary":
        for folder, expected in folders().items():
            print(f"{folder}: {len(list(folder.glob('*-azure-terrified.wav')))}/{expected} WAVs")
        return
    with lock():
        if args.action in ("prepare", "run"):
            for suffix in ("KEY", "REGION"):
                key = f"AZURE_SPEECH_{suffix}"
                if not os.getenv(key) and os.getenv(f"SPEECH_{suffix}"):
                    os.environ[key] = os.environ[f"SPEECH_{suffix}"]
            synthesize(args.action == "prepare")
        if args.action == "e2v":
            for folder in folders():
                subprocess.run([os.getenv("EMOTION_PYTHON", ".venv-emotion/bin/python"), "-u",
                                "classify_cloud_smoke.py", "--device", "cpu", "--path", str(folder)], check=True)
        if args.action != "prepare":
            archive()


if __name__ == "__main__":
    main()
