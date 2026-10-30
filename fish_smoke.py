"""Speak one unchanged question in seven styles with Fish Audio S2-Pro."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from generate import append_jsonl, completed_records, final_answer, read_jsonl, render_preview, write_jsonl

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=ROOT / "examples/tiny.jsonl")
    parser.add_argument("--reference-path", type=Path, default=ROOT / "outputs/manual200-mixed/recordings")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/fish-smoke")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    reference_path = args.reference_path.resolve()
    reference = next(r for r in completed_records(reference_path).values() if r["style"] == "neutral")
    reference_audio = reference_path / reference["audio_path"]
    reference_bytes = reference_audio.read_bytes()
    reference_text = reference["text"]
    question = read_jsonl(args.input_jsonl)[0]
    text = question.get("question", question.get("text"))
    question_id = question.get("id", question.get("question_id", "smoke"))
    styles = json.loads((ROOT / "configs/fish_styles.json").read_text())
    tasks = []
    for style, tags in styles.items():
        clip_id = f"{question_id}__Ryan-reference__{style}__r00"
        tasks.append({
            "question_id": question_id, "clip_id": clip_id, "text": text,
            "answer": final_answer(question["answer"]), "style": style,
            "speaker": "Ryan-reference", "repeat": 0, "seed": 42,
            "model": "fishaudio/s2-pro", "tts_text": f"<|speaker:0|>{tags} {text}",
            "reference_audio": str(reference_audio), "reference_text": reference_text,
            "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
            "audio_path": f"audio/{clip_id}.wav",
        })
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "plan.jsonl", tasks)
    records = completed_records(output)
    render_preview(output, tasks, records)
    if args.prepare_only:
        print(f"Seven-style plan saved: {output / 'plan.jsonl'}")
        return

    import soundfile as sf
    import torch
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

    fish_root = ROOT / "external/fish-speech"
    checkpoint = fish_root / "checkpoints/s2-pro"
    # Keep Fish's internal reference/cache files inside its ignored checkout.
    os.chdir(fish_root)
    precision = torch.bfloat16
    queue = launch_thread_safe_queue(checkpoint, "cuda", precision, compile=False)
    decoder = load_decoder("modded_dac_vq", str(checkpoint / "codec.pth"), "cuda")
    engine = TTSInferenceEngine(queue, decoder, precision, compile=False, model_family="s2")
    voice = ServeReferenceAudio(audio=reference_bytes, text=reference_text, speaker=0)

    for index, task in enumerate(tasks, 1):
        target = output / task["audio_path"]
        if task["clip_id"] in records and target.is_file():
            print(f"[{index}/7] skip {task['style']}", flush=True)
            continue
        print(f"[{index}/7] generate {task['style']}: {task['tts_text']}", flush=True)
        started = time.monotonic()
        request = ServeTTSRequest(
            text=task["tts_text"], references=[voice], seed=task["seed"],
            max_new_tokens=1024, chunk_length=1000, use_memory_cache="on",
        )
        for result in engine.inference(request):
            if result.code == "error":
                raise result.error
            if result.code == "final":
                rate, audio = result.audio
                target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(target, audio, rate, subtype="FLOAT")
                record = dict(task, sample_rate=int(rate), duration_seconds=len(audio) / rate,
                              generation_seconds=time.monotonic() - started,
                              audio_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
                append_jsonl(output / "manifest.jsonl", record)
                records[task["clip_id"]] = record
                render_preview(output, tasks, records)
                print(f"  saved {record['duration_seconds']:.1f}s audio in {record['generation_seconds']:.1f}s", flush=True)
    print(f"Listen: {output / 'listen.html'}")


if __name__ == "__main__":
    main()
