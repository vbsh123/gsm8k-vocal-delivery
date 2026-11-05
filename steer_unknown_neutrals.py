"""Steer held-out neutral-generated, classifier-neutral clips away from unknown."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from scoring import score_response
from steer import make_conditions, read_rows, summarize
from run_steering_followup import install_interventions


def select_neutrals(records, labels, excluded, seed, limit=None):
    latest = {r["clip_id"]: r for r in labels}
    neutral_ids = {clip for clip, r in latest.items() if r["emotion"] == "neutral"}
    selected = {r["question_id"]: r for r in records
                if r["clip_id"] in neutral_ids and r["style"] == "neutral"
                and r["speaker"] == "Ryan" and r["repeat"] == 0
                and r["duration_seconds"] <= 30 and r["question_id"] not in excluded}
    rows = [r for _, r in sorted(selected.items())]
    random.Random(seed).shuffle(rows)
    return rows if limit is None else rows[:limit]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=Path("outputs/manual200-mixed/cropped"))
    p.add_argument("--classifier", choices=["emotion2vec", "sensevoice"], default="emotion2vec")
    p.add_argument("--vectors", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--layers", type=int, nargs="+", default=list(range(26)))
    p.add_argument("--groups", nargs="+", choices=["acoustic", "semantic", "combined", "answer"],
                   default=["acoustic", "answer"])
    p.add_argument("--strength", type=float, default=-1.0)
    p.add_argument("--num-examples", type=int, help="Default: all eligible held-out questions")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args(argv)
    if args.strength >= 0 or not math.isfinite(args.strength):
        p.error("Use a finite negative strength to move away from unknown")
    if args.num_examples is not None and args.num_examples < 1:
        p.error("--num-examples must be positive")
    args.vectors = args.vectors or args.path.parent / f"unknown-vectors-{args.classifier}" / "unknown-vectors.pt"
    args.output = args.output or Path(f"outputs/steer-unknown-{args.classifier}-neutral-L0-25")
    pairs_path = args.vectors.parent / "pairs.jsonl"
    for path in (pairs_path, args.path / "manifest.jsonl",
                 args.path / f"results-{args.classifier}-evaluated.jsonl", args.vectors):
        if not path.is_file():
            p.error(f"Missing required input: {path}")
    excluded = {r["question_id"] for r in read_rows(pairs_path)}
    records = read_rows(args.path / "manifest.jsonl")
    labels = read_rows(args.path / f"results-{args.classifier}-evaluated.jsonl")
    selected = select_neutrals(records, labels, excluded, args.seed, args.num_examples)
    if not selected:
        print("No held-out neutral-generated, classifier-neutral questions. No model loaded.")
        return
    conditions = make_conditions(args.layers, args.groups, [args.strength])
    stat = args.vectors.stat()
    settings = {"classifier": args.classifier, "vectors": str(args.vectors.resolve()),
                "vector_bytes": stat.st_size, "vector_mtime_ns": stat.st_mtime_ns,
                "path": str(args.path.resolve()), "seed": args.seed,
                "mode": "fixed", "conditions": conditions,
                "sign": "unknown minus neutral"}
    plan = {"settings": settings, "recordings": selected, "excluded_question_ids": sorted(excluded)}
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != plan:
            p.error("Selection or settings changed; use a new --output")
    else:
        config_path.write_text(json.dumps(plan, indent=2) + "\n")
    results_path = args.output / "results.jsonl"
    results = read_rows(results_path)
    completed = {(r["clip_id"], r["condition"]) for r in results}
    pending = sum((r["clip_id"], c["name"]) not in completed for r in selected for c in conditions)
    print(f"{len(selected)} held-out neutral-generated AND {args.classifier}-neutral questions.", flush=True)
    print(f"Excluded all {len(excluded)} planned extraction questions.", flush=True)
    print(f"{len(conditions)} conditions: one baseline plus separate layer/group interventions; "
          f"{pending} generations pending. Strength={args.strength}.", flush=True)
    print("Frozen question selection:", config_path, flush=True)
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
    if data["classifier"] != args.classifier or data["sign"] != "unknown minus neutral":
        raise ValueError("Vector classifier or sign does not match this experiment")
    extracted = {r["question_id"] for r in data["pairs"]}
    if any(r["question_id"] in extracted for r in selected):
        raise ValueError("Test selection overlaps vector extraction questions")
    prompt, answer_tokens = data["prompt"], data["max_new_tokens"]
    directions = {}
    for c in conditions[1:]:
        layer, group = c["layer"], c["group"]
        if layer < 0 or layer >= data["pair_counts"]["unknown"].shape[0]:
            raise ValueError(f"Invalid layer: {layer}")
        count = int(data["pair_counts"]["unknown"][layer])
        if count == 0:
            raise ValueError(f"No extraction pairs at layer {layer}")
        vector = data["pair_mean_sums"]["unknown"][layer, data["groups"].index(group)] / count
        if not torch.isfinite(vector).all():
            raise ValueError("Non-finite direction")
        directions[layer, group] = vector.to(device="cuda", dtype=torch.bfloat16)
    del data
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
            if len(audio) / rate > 30:
                raise ValueError(f"Audio exceeds 30 seconds: {r['clip_id']}")
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
                interventions = [] if c["name"] == "baseline" else [
                    {"layer": c["layer"], "group": c["group"], "strength": c["strength"]}]
                print(f"[{index}/{len(selected)}] [{c_index}/{len(conditions)}] "
                      f"{r['clip_id']} {c['name']}", flush=True)
                with install_interventions(layers, directions, interventions, bounds, answer_tokens) as states:
                    with torch.inference_mode():
                        response = model.generate(
                            tokenizer, input_audio_features=inputs.input_features,
                            input_audio_mask=inputs.attention_mask, system_prompt="",
                            input_texts=[prompt], steps=128, block_length=32,
                            max_new_tokens=answer_tokens, temperature=0.0, accelerate="fast_dllm")
                row = {"clip_id": r["clip_id"], "question_id": r["question_id"],
                       "condition": c["name"], "layer": c["layer"], "group": c["group"],
                       "strength": c["strength"], "classifier": args.classifier,
                       "vector_style": "unknown", "mode": "fixed", "seed": seed,
                       "response": response, "prompt": prompt, "max_new_tokens": answer_tokens,
                       "decoder": "fast_dllm", "forwards": states[0]["state"]["forwards"],
                       "steered_forwards": states[0]["state"]["steered"],
                       **score_response(response, r["answer"])}
                with results_path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                results.append(row)
                completed.add((r["clip_id"], c["name"]))
                print(f"  correct={row['correct']} answer={row['predicted_answer']} "
                      f"steered_forwards={row['steered_forwards']}", flush=True)
    finally:
        decoder.insert_audio_embedding = original_insert
    summary = summarize(results, conditions)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
