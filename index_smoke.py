"""Generate selected questions with IndexTTS-2.5's native emotion controls."""

import argparse
import hashlib
import os
from pathlib import Path
import random
import time

from generate import append_jsonl, completed_records, final_answer, read_jsonl, render_preview, write_jsonl

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=ROOT / "examples/tiny.jsonl")
    parser.add_argument("--reference-path", type=Path, default=ROOT / "outputs/manual200-mixed/recordings")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/index-smoke")
    parser.add_argument("--intensity", type=float, default=0.8)
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--styles", nargs="+", choices=["neutral", "happy", "angry", "sad", "afraid", "disgusted"],
                        default=["neutral", "happy", "angry", "sad", "afraid", "disgusted"])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    reference_path = args.reference_path.resolve()
    reference = next(r for r in completed_records(reference_path).values() if r["style"] == "neutral")
    reference_audio = reference_path / reference["audio_path"]
    reference_hash = hashlib.sha256(reference_audio.read_bytes()).hexdigest()
    questions = read_jsonl(args.input_jsonl)[:args.num_examples]
    strength = args.intensity
    # Official order: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm.
    vectors = {
        "neutral": None,
        "happy": [strength, 0, 0, 0, 0, 0, 0, 0],
        "angry": [0, strength, 0, 0, 0, 0, 0, 0],
        "sad": [0, 0, strength, 0, 0, 0, 0, 0],
        "afraid": [0, 0, 0, strength, 0, 0, 0, 0],
        "disgusted": [0, 0, 0, 0, strength, 0, 0, 0],
    }
    tasks = []
    for question_index, question in enumerate(questions):
        text = question.get("question", question.get("text"))
        question_id = question.get("id", question.get("question_id", f"local_{question_index:05d}"))
        for style in args.styles:
            clip_id = f"{question_id}__Ryan-reference__{style}__r00"
            tasks.append({
                "question_id": question_id, "clip_id": clip_id, "text": text,
                "answer": final_answer(question["answer"]), "style": style,
                "speaker": "Ryan-reference", "repeat": 0, "seed": 42,
                "model": "IndexTeam/IndexTTS-2.5", "emotion_vector": vectors[style],
                "reference_audio": str(reference_audio), "reference_sha256": reference_hash,
                "audio_path": f"audio/{clip_id}.wav",
            })
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "plan.jsonl", tasks)
    records = completed_records(output)
    render_preview(output, tasks, records)
    if args.prepare_only:
        print(f"{len(tasks)}-condition plan saved: {output / 'plan.jsonl'}")
        return

    import numpy as np
    import soundfile as sf
    import torch

    index_root = ROOT / "external/index-tts"
    os.chdir(index_root)
    from indextts.infer_v2_5 import IndexTTS2

    checkpoint = index_root / "checkpoints"
    tts = IndexTTS2(
        cfg_path=str(checkpoint / "config.yaml"), model_dir=str(checkpoint),
        device="cuda:0", use_bf16=True, use_cuda_kernel=False, use_qwen_emo=False,
    )
    for index, task in enumerate(tasks, 1):
        target = output / task["audio_path"]
        if task["clip_id"] in records and target.is_file():
            print(f"[{index}/{len(tasks)}] skip {task['style']}", flush=True)
            continue
        print(f"[{index}/{len(tasks)}] generate {task['style']}: {task['emotion_vector']}", flush=True)
        random.seed(task["seed"])
        np.random.seed(task["seed"])
        torch.manual_seed(task["seed"])
        torch.cuda.manual_seed_all(task["seed"])
        target.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        tts.infer(
            spk_audio_prompt=str(reference_audio), text=task["text"], lang="EN",
            output_path=str(target), emo_vector=task["emotion_vector"],
            use_random=False,
        )
        info = sf.info(target)
        record = dict(task, sample_rate=int(info.samplerate), duration_seconds=info.duration,
                      generation_seconds=time.monotonic() - started,
                      audio_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        append_jsonl(output / "manifest.jsonl", record)
        records[task["clip_id"]] = record
        render_preview(output, tasks, records)
        print(f"  saved {record['duration_seconds']:.1f}s audio in {record['generation_seconds']:.1f}s", flush=True)
    print(f"Listen: {output / 'listen.html'}")


if __name__ == "__main__":
    main()
