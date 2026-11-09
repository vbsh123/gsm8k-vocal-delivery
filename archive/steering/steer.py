"""Historical steering CLI with the original per-recording derived seeds."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from scoring import score_response
from steer import read_rows, select_neutrals, make_conditions, reuse_questions, make_steering_hook, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True, help="Cropped recordings folder")
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--exclude-results", type=Path, nargs="+", required=True,
                        help="Exclude ALL question IDs in these JSONLs: extraction labels, and optionally prior steering runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions-from", type=Path,
                        help="Reuse exactly the recordings/order from a previous run_config.json")
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--style", choices=["angry", "happy", "confident"], default="angry")
    parser.add_argument("--layers", nargs="+", type=int, default=[15], help="Zero-based layer indices, tested separately")
    parser.add_argument("--groups", nargs="+", choices=["acoustic", "semantic", "combined", "answer"],
                        default=["acoustic", "answer"], help="Position groups, tested separately")
    parser.add_argument("--strengths", nargs="+", type=float, default=[0, 0.5, -0.5],
                        help="Multipliers of raw mean differences; baseline is always included once")
    parser.add_argument("--mode", choices=["fixed", "per-pass"], default="fixed")
    parser.add_argument("--min-pairs", type=int, default=1, help="Minimum contributors for a per-pass direction")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.num_examples <= 0 or args.min_pairs <= 0:
        parser.error("num-examples and min-pairs must be positive")
    if not all(math.isfinite(a) for a in args.strengths):
        parser.error("strengths must be finite")
    if not all(p.is_file() for p in args.exclude_results):
        parser.error("exclude-results must be the existing extraction classification file")
    conditions = make_conditions(args.layers, args.groups, args.strengths)
    stat = args.vectors.stat()
    settings = dict(path=str(args.path.resolve()), vectors=str(args.vectors.resolve()),
                    vector_bytes=stat.st_size, vector_mtime_ns=stat.st_mtime_ns,
                    exclude_results=[str(p.resolve()) for p in args.exclude_results],
                    num_examples=args.num_examples, seed=args.seed, style=args.style,
                    mode=args.mode, min_pairs=args.min_pairs, conditions=conditions)
    if args.questions_from:
        settings["questions_from"] = str(args.questions_from.resolve())
        settings["questions_from_sha256"] = hashlib.sha256(args.questions_from.read_bytes()).hexdigest()
    config_path = args.output / "run_config.json"
    records = {r["clip_id"]: r for r in read_rows(args.path / "manifest.jsonl")}
    excluded = {r["question_id"] for p in args.exclude_results for r in read_rows(p)}
    if config_path.exists():
        plan = json.loads(config_path.read_text())
        if plan["settings"] != settings:
            parser.error("Run settings changed; use a different --output folder")
        selected = plan["recordings"]
        for r in selected:
            if (records.get(r["clip_id"]) != r or r["question_id"] in excluded):
                parser.error("A selected recording changed or entered the exclusion set; use a new output")
    else:
        if args.questions_from:
            previous = json.loads(args.questions_from.read_text())["recordings"]
            selected = reuse_questions(previous, records, excluded, args.num_examples)
        else:
            selected = select_neutrals(records.values(), excluded, args.num_examples, args.seed)
        plan = dict(settings=settings, recordings=selected, excluded_question_ids=sorted(excluded))
        args.output.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(plan, indent=2) + "\n")
    results_path = args.output / "results.jsonl"
    results = read_rows(results_path)
    completed = {(r["clip_id"], r["condition"]) for r in results}
    pending = sum((r["clip_id"], c["name"]) not in completed for r in selected for c in conditions)
    print(f"{len(selected)} unseen neutral questions × {len(conditions)} conditions; "
          f"{pending} generations remain. Frozen plan: {config_path}", flush=True)
    if args.prepare_only:
        return
    if not pending:
        summarize(results, conditions)
        return

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    import src.fast_dllm.diffa_generate as decoder
    from extract_vectors import load_model

    data = torch.load(args.vectors, map_location="cpu", weights_only=True, mmap=True)
    prompt = data["prompt"]
    answer_tokens = data["max_new_tokens"]
    if data.get("sign") != "style minus neutral":
        raise ValueError("Unknown vector sign convention")
    vectors = {}
    for c in conditions[1:]:
        layer, group = c["layer"], data["groups"].index(c["group"])
        if layer < 0 or layer >= data["sums"][args.style].shape[0]:
            raise ValueError(f"Invalid layer index: {layer}")
        if args.mode == "fixed":
            count = int(data["pair_counts"][args.style][layer])
            if not count:
                raise ValueError(f"No extraction pairs at layer {layer}")
            vector = data["pair_mean_sums"][args.style][layer, group] / count
            valid = None
        else:
            counts = data["counts"][args.style][layer]
            vector = data["sums"][args.style][layer, :, group] / counts.clamp_min(1)[:, None]
            valid = counts >= args.min_pairs
            if not valid.any():
                raise ValueError(f"No supported passes at layer {layer}")
        if not torch.isfinite(vector).all():
            raise ValueError("Vector contains non-finite values")
        vectors[c["name"]] = (vector, valid)

    model, tokenizer, feature_extractor = load_model()
    layers = model.llm_model.model.transformer.blocks
    bounds = {}
    original_insert = decoder.insert_audio_embedding

    def capture_insert(text_embeds, speech_embeds, audio_info_lengths):
        start = int(audio_info_lengths[0].item())
        bounds.update(start=start, acoustic_end=start+model.config.prompt_size,
                      end=start+speech_embeds.shape[1])
        return original_insert(text_embeds, speech_embeds, audio_info_lengths)

    decoder.insert_audio_embedding = capture_insert
    try:
        for index, r in enumerate(selected, 1):
            if all((r["clip_id"], c["name"]) in completed for c in conditions):
                continue
            audio, rate = sf.read(args.path / r["audio_path"], dtype="float32")
            if len(audio) > 30 * rate:
                raise ValueError(f"Actual WAV exceeds 30 seconds: {r['clip_id']}")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if rate != 16000:
                audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
            inputs = feature_extractor(audio, sampling_rate=16000, truncation=False,
                                       return_tensors="pt", return_attention_mask=True).to("cuda")
            for c_index, c in enumerate(conditions, 1):
                if (r["clip_id"], c["name"]) in completed:
                    continue
                seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{r['clip_id']}".encode()).digest()[:4], "big")
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                bounds.clear()
                if c["name"] == "baseline":
                    hook, state = make_steering_hook(None, None, "fixed", "answer", 0,
                                                     bounds, answer_tokens)
                    handle = layers[0].register_forward_hook(hook)
                else:
                    vector, valid = vectors[c["name"]]
                    vector = vector.to(device="cuda", dtype=torch.bfloat16)
                    hook, state = make_steering_hook(vector, valid, args.mode, c["group"],
                                                     c["strength"], bounds, answer_tokens)
                    handle = layers[c["layer"]].register_forward_hook(hook)
                print(f"[{index}/{len(selected)}] [{c_index}/{len(conditions)}] "
                      f"{r['clip_id']} {c['name']}", flush=True)
                try:
                    with torch.inference_mode():
                        response = model.generate(
                            tokenizer, input_audio_features=inputs.input_features,
                            input_audio_mask=inputs.attention_mask, system_prompt="",
                            input_texts=[prompt], steps=128, block_length=32,
                            max_new_tokens=answer_tokens, temperature=0.0, accelerate="fast_dllm")
                finally:
                    handle.remove()
                result = dict(clip_id=r["clip_id"], question_id=r["question_id"],
                              condition=c["name"], layer=c["layer"], group=c["group"],
                              strength=c["strength"], vector_style=args.style, mode=args.mode,
                              seed=seed, response=response, prompt=prompt,
                              max_new_tokens=answer_tokens, decoder="fast_dllm",
                              forwards=state["forwards"],
                              steered_forwards=state["steered"],
                              **score_response(response, r["answer"]))
                with results_path.open("a") as f:
                    f.write(json.dumps(result) + "\n")
                results.append(result)
                completed.add((r["clip_id"], c["name"]))
                print(f"  correct={result['correct']} answer={result['predicted_answer']} "
                      f"steered_forwards={result['steered_forwards']}", flush=True)
    finally:
        decoder.insert_audio_embedding = original_insert
    summary = summarize(results, conditions)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
