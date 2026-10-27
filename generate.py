#!/usr/bin/env python3
"""Generate paired vocal deliveries of GSM8K questions with Qwen3-TTS.

Planning, preview generation, and local JSONL input use only the standard library.
GPU dependencies are loaded only when synthesis is needed.
"""

import argparse
import hashlib
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time


ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
READ_ONLY = (
    " Speak only the supplied text, exactly as written. Do not answer the question, "
    "explain it, add an introduction, or say these instructions. Preserve all numbers and words."
)


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}: {exc}") from exc
    return rows


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Use only letters, digits, underscores, or hyphens in IDs/names: {value!r}")
    return value


def final_answer(answer):
    return str(answer).rsplit("####", 1)[-1].strip()


def load_questions(args):
    if args.input_jsonl:
        rows = read_jsonl(args.input_jsonl)
        source = {"dataset": "local_jsonl", "path": str(args.input_jsonl.resolve())}
    else:
        from datasets import load_dataset

        dataset = load_dataset("openai/gsm8k", "main", split=args.split,
                               revision=args.dataset_revision)
        rows = [dict(row, id=f"gsm8k_{args.split}_{i:05d}") for i, row in enumerate(dataset)]
        source = {"dataset": "openai/gsm8k", "config": "main", "split": args.split,
                  "revision": args.dataset_revision, "fingerprint": dataset._fingerprint}
    clean = []
    seen = set()
    for index, row in enumerate(rows):
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question.strip() or answer is None:
            raise ValueError(f"Question row {index} needs nonempty 'question' and an 'answer'.")
        question_id = safe_name(str(row.get("id", f"local_{index:05d}")))
        if question_id in seen:
            raise ValueError(f"Duplicate question ID: {question_id}")
        seen.add(question_id)
        if args.max_words and len(question.split()) > args.max_words:
            continue
        clean.append({"question_id": question_id, "text": question,
                      "answer": final_answer(answer), "source_answer": str(answer)})
    # Shuffling the complete index list makes a larger requested sample extend
    # the same prefix, unlike random.sample(..., k) with changing k.
    random.Random(args.seed).shuffle(clean)
    if args.num_examples > len(clean):
        raise ValueError(f"Requested {args.num_examples} questions; only {len(clean)} available after filtering.")
    return clean[:args.num_examples], source


def make_plan(args, questions, styles):
    tasks = []
    for question in questions:
        for speaker in args.speakers:
            safe_name(speaker)
            for repeat in range(args.repeats):
                # Match the initial RNG seed across styles within a paired set.
                seed_key = f"{args.seed}/{question['question_id']}/{speaker}/{repeat}"
                generation_seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
                for style in args.styles:
                    safe_name(style)
                    if style not in styles or not isinstance(styles[style], str) or not styles[style].strip():
                        raise ValueError(f"Missing or invalid style prompt: {style}")
                    clip_id = f"{question['question_id']}__{speaker}__{style}__r{repeat:02d}"
                    tasks.append(dict(question, clip_id=clip_id, speaker=speaker,
                                      style=style, repeat=repeat, seed=generation_seed,
                                      instruction=styles[style] + READ_ONLY,
                                      audio_path=f"audio/{clip_id}.wav"))
    return tasks


def completed_records(output):
    records = {}
    for record in read_jsonl(output / "manifest.jsonl"):
        records[record["clip_id"]] = record
    return records


def render_preview(output, tasks, records):
    esc = lambda value: html.escape(str(value), quote=True)
    chunks = ["""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GSM8K vocal delivery listening page</title>
<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}
article{border-top:1px solid #ccc;padding:20px 0}section{display:flex;flex-wrap:wrap;gap:16px}
.clip{background:#f3f5f8;padding:12px;border-radius:8px;width:310px}
audio{width:100%}small{display:block;color:#555}summary{cursor:pointer}</style>
<h1>GSM8K vocal delivery</h1><p>Compare wording, numbers, completeness, and perceived delivery.
Style labels indicate requested delivery; they are not verified annotations.</p>"""]
    previous = None
    for task in tasks:
        if task["question_id"] != previous:
            if previous is not None:
                chunks.append("</section></article>")
            chunks.append(f"<article><h2>{esc(task['question_id'])}</h2><p>{esc(task['text'])}</p>"
                          f"<details><summary>Reference answer (never sent to TTS)</summary>"
                          f"{esc(task['answer'])}</details><section>")
            previous = task["question_id"]
        record = records.get(task["clip_id"])
        ready = record is not None and (output / task["audio_path"]).is_file()
        player = (f'<audio controls preload="none" src="{esc(task["audio_path"])}"></audio>'
                  if ready else "<p>Not generated yet</p>")
        chunks.append(f'<div class="clip"><b>{esc(task["style"])}</b>'
                      f'<small>{esc(task["speaker"])} · repeat {task["repeat"]}</small>{player}</div>')
    if previous is not None:
        chunks.append("</section></article>")
    chunks.append("</html>")
    (output / "listen.html").write_text("\n".join(chunks), encoding="utf-8")


class QwenBackend:
    def __init__(self, args):
        import torch
        import soundfile as sf
        from qwen_tts import Qwen3TTSModel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not detected. Use a CUDA-enabled PyTorch install; planning needs no GPU.")
        self.torch, self.sf, self.args = torch, sf, args
        self.model = Qwen3TTSModel.from_pretrained(
            MODEL, revision=args.model_revision, device_map="cuda:0",
            dtype=torch.bfloat16, attn_implementation=args.attn)
        supported = self.model.get_supported_speakers()
        if supported:
            bad = [s for s in args.speakers if s.lower() not in {x.lower() for x in supported}]
            if bad:
                raise ValueError(f"Unsupported speakers {bad}; available: {supported}")
        self.info = {"gpu": torch.cuda.get_device_name(0), "packages": {}}
        for package in ("qwen-tts", "torch", "transformers", "datasets", "soundfile"):
            self.info["packages"][package] = importlib.metadata.version(package)
        self.info["resolved_model_revision"] = getattr(self.model.model.config, "_commit_hash", None)

    def generate(self, task, target):
        import numpy as np

        torch = self.torch
        random.seed(task["seed"])
        np.random.seed(task["seed"])
        torch.manual_seed(task["seed"])
        torch.cuda.manual_seed_all(task["seed"])
        with torch.inference_mode():
            # Deliberately pass neither answer nor source_answer to the TTS model.
            wavs, rate = self.model.generate_custom_voice(
                text=task["text"], language="English", speaker=task["speaker"],
                instruct=task["instruction"], do_sample=True,
                temperature=self.args.temperature, max_new_tokens=self.args.max_new_tokens)
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if rate <= 0 or audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("TTS returned empty or non-finite audio or an invalid sample rate.")
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < 1e-7:
            raise ValueError("TTS returned effectively silent audio.")
        tmp = target.with_suffix(".tmp.wav")
        # Float WAV preserves amplitude without clipping, normalization or resampling.
        self.sf.write(tmp, audio, rate, format="WAV", subtype="FLOAT")
        tmp.replace(target)
        duration = len(audio) / rate
        return {"sample_rate": int(rate), "duration_seconds": duration,
                "peak_amplitude": float(np.max(np.abs(audio))), "rms": rms,
                # 12 Hz codes: heuristic only; listen / ASR-check for truncation.
                "possible_token_limit": duration >= self.args.max_new_tokens / 12 * 0.95}


def run_tasks(output, tasks, backend):
    records = completed_records(output)
    failures = 0
    try:
        for index, task in enumerate(tasks, 1):
            target = output / task["audio_path"]
            if task["clip_id"] in records and target.is_file() and target.stat().st_size > 44:
                print(f"[{index}/{len(tasks)}] skip {task['clip_id']}", flush=True)
                continue
            print(f"[{index}/{len(tasks)}] generate {task['clip_id']}", flush=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            try:
                metrics = backend.generate(task, target)
                if not target.is_file() or target.stat().st_size <= 44:
                    raise ValueError("Audio file was not written correctly.")
                record = dict(task, **metrics, generation_seconds=time.monotonic() - started)
                record["audio_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
                append_jsonl(output / "manifest.jsonl", record)
                records[task["clip_id"]] = record
            except Exception as exc:
                failures += 1
                append_jsonl(output / "errors.jsonl", {"clip_id": task["clip_id"],
                                                      "error": str(exc), "time": time.time()})
                print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
    finally:
        render_preview(output, tasks, records)
    print(f"Finished: {len(tasks) - failures}/{len(tasks)} clips complete; {failures} failed this run.")
    return 1 if failures else 0


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--num-examples", type=int, default=10, help="Number of distinct questions")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--seed", type=int, default=42, help="Question shuffle and generation seed")
    p.add_argument("--speakers", nargs="+", default=["Ryan"], help="Fixed Qwen voices; English: Ryan Aiden")
    p.add_argument("--styles", nargs="+", default=["neutral", "confident", "hesitant", "happy", "angry", "sad", "stressed"])
    p.add_argument("--repeats", type=int, default=1, help="Independent renders per question/voice/style")
    p.add_argument("--styles-file", type=Path, default=ROOT / "configs/styles.json")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/pilot")
    p.add_argument("--input-jsonl", type=Path, help="Local rows with question, answer, optional id; bypasses GSM8K download")
    p.add_argument("--max-words", type=int, default=0, help="Filter long questions before sampling; 0 disables")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=2048, help="Speech-code token cap; inspect clips for truncation")
    p.add_argument("--attn", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")
    p.add_argument("--model-revision", default="main", help="HF revision; pin a commit for reproducibility")
    p.add_argument("--dataset-revision", default="main", help="HF revision; pin a commit for reproducibility")
    p.add_argument("--prepare-only", "--dry-run", action="store_true", help="Save plan and listening page without loading TTS")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if min(args.num_examples, args.repeats, args.max_new_tokens) < 1 or args.max_words < 0:
        p.error("Example/repeat/token counts must be positive; max-words must be nonnegative.")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        p.error("temperature must be finite and positive")
    for name in ("speakers", "styles"):
        values = getattr(args, name)
        if len(values) != len(set(v.lower() for v in values)):
            p.error(f"Duplicate {name} would overwrite paired clips.")
    try:
        styles = json.loads(args.styles_file.read_text(encoding="utf-8"))
        if not isinstance(styles, dict):
            raise ValueError("styles-file must be a JSON object mapping style names to prompts.")
        # Accept the common misspelling while using the canonical label on disk.
        args.styles = ["hesitant" if s == "hesistant" else s for s in args.styles]
        if len(args.styles) != len(set(args.styles)):
            raise ValueError("Duplicate styles after normalizing 'hesistant' to 'hesitant'.")
        questions, source = load_questions(args)
        tasks = make_plan(args, questions, styles)
        config = {"schema_version": 1, "model": MODEL, "model_revision": args.model_revision,
                  "source": source, "seed": args.seed, "temperature": args.temperature,
                  "max_new_tokens": args.max_new_tokens, "attention": args.attn,
                  "language": "English", "wav_subtype": "FLOAT", "tasks": tasks}
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        config_path = output / "run_config.json"
        if config_path.exists():
            if json.loads(config_path.read_text(encoding="utf-8")) != config:
                raise ValueError("Output directory contains a different experiment. Use a new --output directory.")
        elif any(output.iterdir()):
            raise ValueError("Output directory is not empty and has no run_config.json; use a fresh --output.")
        else:
            write_json(config_path, config)
        write_jsonl(output / "questions.jsonl", questions)
        write_jsonl(output / "plan.jsonl", tasks)
        records = completed_records(output)
        render_preview(output, tasks, records)
        print(f"{len(questions)} questions × {len(args.speakers)} voices × {len(args.styles)} styles "
              f"× {args.repeats} repeats = {len(tasks)} clips\nOutput: {output}", flush=True)
        if args.prepare_only:
            print("Plan saved; TTS was not loaded. Rerun without --prepare-only to generate.")
            return 0
        pending = [t for t in tasks if t["clip_id"] not in records
                   or not (output / t["audio_path"]).is_file()
                   or (output / t["audio_path"]).stat().st_size <= 44]
        if not pending:
            print("All clips already complete; TTS was not loaded.")
            return 0
        backend = QwenBackend(args)
        write_json(output / "environment.json", backend.info)
        return run_tasks(output, tasks, backend)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
