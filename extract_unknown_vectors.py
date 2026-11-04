"""Extract classifier-unknown minus classifier-neutral directions from matched audio."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


PROMPT = (
    "Solve the math problem in the audio. Give concise step-by-step reasoning "
    "with only the necessary calculations. If the question asks for a total, "
    "compute that total. End with one plain-text line in exactly this format: "
    "Final answer: <number>. Put a single numeric answer on that line, "
    "without Markdown, units, or separate component counts."
)
GROUPS = ["acoustic", "semantic", "combined", "answer"]
UNKNOWN_TAGS = {"emotion2vec": "<unk>", "sensevoice": "emo_unknown"}


def read_rows(path):
    with path.open() as stream:
        return {r["clip_id"]: r for line in stream if line.strip() for r in [json.loads(line)]}


def select_pairs(manifest, predictions, classifier, seed):
    groups = defaultdict(lambda: {"neutral": [], "unknown": []})
    for clip_id, prediction in predictions.items():
        if clip_id not in manifest:
            continue
        emotion = prediction["emotion"]
        label = "neutral" if emotion == "neutral" else "unknown" if emotion == UNKNOWN_TAGS[classifier] else None
        if label is None:
            continue
        r = manifest[clip_id]
        groups[r["question_id"], r["speaker"], r["repeat"]][label].append(clip_id)
    pairs = []
    for key, labels in sorted(groups.items()):
        if not labels["neutral"] or not labels["unknown"]:
            continue
        pair_seed = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:4], "big")
        rng = random.Random(pair_seed)
        pairs.append({"question_id": key[0], "speaker": key[1], "repeat": key[2],
                      "neutral_id": rng.choice(sorted(labels["neutral"])),
                      "unknown_id": rng.choice(sorted(labels["unknown"])), "seed": pair_seed})
    return pairs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=Path("outputs/manual200-mixed/cropped"))
    p.add_argument("--classifier", choices=tuple(UNKNOWN_TAGS), default="emotion2vec")
    p.add_argument("--output", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    if args.max_new_tokens < 32 or args.max_new_tokens % 32 or 128 % (args.max_new_tokens // 32):
        p.error("--max-new-tokens must be a multiple of 32 with its block count dividing 128")
    output = args.output or args.path.parent / f"unknown-vectors-{args.classifier}"
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_rows(args.path / "manifest.jsonl")
    predictions = read_rows(args.path / f"results-{args.classifier}-evaluated.jsonl")
    pair_path = output / "pairs.jsonl"
    if pair_path.exists():
        pairs = [json.loads(line) for line in pair_path.read_text().splitlines() if line.strip()]
    else:
        pairs = select_pairs(manifest, predictions, args.classifier, args.seed)
        pair_path.write_text("".join(json.dumps(r) + "\n" for r in pairs))
    all_pairs = pairs
    if args.limit is not None:
        pairs = pairs[:args.limit]
    print(f"{args.classifier}: {len(pairs)} matched pairs, {2*len(pairs)} generations.", flush=True)
    print("Labels are classifier predictions; both sides may come from any generated style.", flush=True)
    print("Direction: unknown minus neutral. Negative strength points away from unknown.", flush=True)
    print("Selected pairs:", pair_path, flush=True)
    if args.prepare_only or not pairs:
        return

    import numpy as np
    import soundfile as sf
    import torch
    import src.fast_dllm.diffa_generate as decoder
    from extract_vectors import load_model, gen

    vectors_path = output / "unknown-vectors.pt"
    if vectors_path.exists():
        state = torch.load(vectors_path, map_location="cpu", weights_only=True)
        if (state["classifier"] != args.classifier or state["max_new_tokens"] != args.max_new_tokens
                or state["seed"] != args.seed or state["pairs"] != all_pairs):
            p.error("Saved extraction settings differ; use a new --output")
        done = set(state["completed_neutral_ids"])
    else:
        state = None
        done = set()
    pending = [pair for pair in pairs if pair["neutral_id"] not in done]
    if not pending:
        print("All selected pairs already extracted.", flush=True)
        return
    print(f"{len(pending)} pairs pending; loading DIFFA.", flush=True)
    model, tokenizer, feature_extractor = load_model()
    layers = model.llm_model.model.transformer.blocks
    max_passes = args.max_new_tokens
    if state is None:
        hidden_size = model.llm_model.get_input_embeddings().weight.shape[1]
        state = {
            "sums": {"unknown": torch.zeros(len(layers), max_passes, len(GROUPS), hidden_size)},
            "counts": {"unknown": torch.zeros(len(layers), max_passes, dtype=torch.long)},
            "pair_mean_sums": {"unknown": torch.zeros(len(layers), len(GROUPS), hidden_size)},
            "pair_counts": {"unknown": torch.zeros(len(layers), dtype=torch.long)},
            "neutral_mean_sum": torch.zeros(len(layers), len(GROUPS), hidden_size),
            "unknown_mean_sum": torch.zeros(len(layers), len(GROUPS), hidden_size),
            "groups": GROUPS, "styles": ["unknown"], "max_passes": max_passes,
            "prompt": PROMPT, "max_new_tokens": args.max_new_tokens, "steps": 128,
            "classifier": args.classifier, "sign": "unknown minus neutral", "seed": args.seed,
            "pairs": all_pairs,
            "completed_neutral_ids": [],
        }
    bounds = {}
    captured = []
    original_insert = decoder.insert_audio_embedding

    def capture_insert(text_embeds, speech_embeds, audio_info_lengths):
        start = int(audio_info_lengths[0].item())
        bounds.update(start=start, acoustic_end=start+model.config.prompt_size,
                      end=start+speech_embeds.shape[1])
        return original_insert(text_embeds, speech_embeds, audio_info_lengths)

    def make_hook(layer_index):
        def hook(module, inputs, result):
            hidden = result[0]
            acoustic = hidden[0, bounds["start"]:bounds["acoustic_end"]]
            semantic = hidden[0, bounds["acoustic_end"]:bounds["end"]]
            combined = hidden[0, bounds["start"]:bounds["end"]]
            answer = hidden[0, -args.max_new_tokens:]
            captured[layer_index].append(torch.stack([
                values.float().mean(dim=0) for values in (acoustic, semantic, combined, answer)
            ]).cpu())
        return hook

    def capture(clip_id, seed):
        nonlocal captured
        captured = [[] for _ in layers]
        bounds.clear()
        audio, rate = sf.read(args.path / manifest[clip_id]["audio_path"], dtype="float32")
        if len(audio) / rate > 30:
            raise ValueError(f"Clip exceeds Whisper's 30-second limit: {clip_id}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        response = gen(audio, rate, model, tokenizer, feature_extractor, PROMPT, args.max_new_tokens)
        summary = torch.stack([torch.stack(passes) for passes in captured])
        captured = []
        return response, summary

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    decoder.insert_audio_embedding = capture_insert
    try:
        for index, pair in enumerate(pending, 1):
            print(f"[{index}/{len(pending)}] {pair['question_id']} "
                  f"neutral={pair['neutral_id']} unknown={pair['unknown_id']}", flush=True)
            neutral_response, neutral = capture(pair["neutral_id"], pair["seed"])
            unknown_response, unknown = capture(pair["unknown_id"], pair["seed"])
            n = min(neutral.shape[1], unknown.shape[1])
            if n > max_passes:
                raise ValueError(f"{n} forwards exceeds allocated capacity {max_passes}")
            difference = unknown[:, :n] - neutral[:, :n]
            state["sums"]["unknown"][:, :n] += difference
            state["counts"]["unknown"][:, :n] += 1
            state["pair_mean_sums"]["unknown"] += difference.mean(dim=1)
            state["pair_counts"]["unknown"] += 1
            state["neutral_mean_sum"] += neutral[:, :n].mean(dim=1)
            state["unknown_mean_sum"] += unknown[:, :n].mean(dim=1)
            state["completed_neutral_ids"].append(pair["neutral_id"])
            with (output / "responses.jsonl").open("a") as stream:
                stream.write(json.dumps({**pair, "neutral_response": neutral_response,
                    "unknown_response": unknown_response, "neutral_forwards": neutral.shape[1],
                    "unknown_forwards": unknown.shape[1], "matched_forwards": n}) + "\n")
            print(f"  forwards neutral={neutral.shape[1]} unknown={unknown.shape[1]} matched={n}", flush=True)
            del neutral, unknown, difference
    finally:
        for handle in handles:
            handle.remove()
        decoder.insert_audio_embedding = original_insert
        temporary = vectors_path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(vectors_path)
        print("Saved:", vectors_path, flush=True)


if __name__ == "__main__":
    main()
