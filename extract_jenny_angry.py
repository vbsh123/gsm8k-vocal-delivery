"""Extract angry-minus-neutral directions from matched emotion2vec-agreed Jenny audio."""

import argparse
import json
from pathlib import Path
import random
import wave

from compare_cloud_accuracy import PROMPT, read


GROUPS = ["acoustic", "semantic", "combined", "answer"]


def recordings(paths):
    records, predictions = {}, {}
    for folder in paths:
        labels = {r["audio_path"]: r for r in read(folder / "results-emotion2vec.jsonl")}
        for task in read(folder / "plan.jsonl"):
            if task["provider"] != "azure" or task["style"] not in ("neutral", "angry"):
                continue
            key = (task["question_id"], task["voice"], task["style"])
            audio_path = (folder / task["audio_path"]).resolve()
            with wave.open(str(audio_path), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
            clip_id = f'{task["question_id"]}__{task["voice"]}__{task["style"]}__r00'
            records[key] = {"clip_id": clip_id, "question_id": task["question_id"],
                            "style": task["style"], "speaker": task["voice"], "repeat": 0,
                            "audio_path": str(audio_path), "duration_seconds": duration,
                            "answer": str(task["answer"]).rsplit("####", 1)[-1].strip()}
            predictions[clip_id] = labels.get(task["audio_path"], {}).get("emotion")
    return records, predictions


def select_pairs(records, predictions, seed):
    if seed != 42:
        raise ValueError("Azure extraction seed is fixed at 42")
    pairs = []
    for (qid, voice, style), neutral in sorted(records.items()):
        if style != "neutral":
            continue
        angry = records.get((qid, voice, "angry"))
        if not angry or max(neutral["duration_seconds"], angry["duration_seconds"]) > 30:
            continue
        if predictions.get(neutral["clip_id"]) != "neutral" or predictions.get(angry["clip_id"]) != "angry":
            continue
        pairs.append({"question_id": qid, "speaker": voice, "neutral_id": neutral["clip_id"],
                      "angry_id": angry["clip_id"], "seed": 42})
    return pairs


def freeze(path, rows):
    if path.exists() and read(path) != rows:
        raise ValueError(f"Saved selection changed; use another output folder: {path}")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", type=Path, nargs="+", default=[
        Path("outputs/azure-jenny500-extra"), Path("outputs/azure-jenny500-neutral")])
    p.add_argument("--output", type=Path, default=Path("outputs/jenny-angry-vectors"))
    p.add_argument("--seed", type=int, choices=[42], default=42, help="Generation seed is fixed at 42.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, help="Process only the first N pairs, for a smoke test.")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args(argv)
    records, predictions = recordings(args.paths)
    manifest = {row["clip_id"]: row for row in records.values()}
    all_pairs = select_pairs(records, predictions, args.seed)
    excluded = [{"question_id": qid} for qid in sorted({r["question_id"] for r in manifest.values()})]
    args.output.mkdir(parents=True, exist_ok=True)
    freeze(args.output / "pairs.jsonl", all_pairs)
    freeze(args.output / "extraction-questions.jsonl", excluded)
    freeze(args.output / "manifest.jsonl", list(manifest.values()))
    pairs = all_pairs if args.limit is None else all_pairs[:args.limit]
    print(f"{len(pairs)} matched, emotion2vec-agreed angry/neutral pairs; {2*len(pairs)} generations.", flush=True)
    print("Direction: angry minus neutral. Selection does not use reasoning correctness.", flush=True)
    if args.prepare_only or not pairs:
        return

    import numpy as np
    import soundfile as sf
    import torch
    import src.fast_dllm.diffa_generate as decoder
    from extract_vectors import load_model, gen

    vector_path = args.output / "style-vectors.pt"
    state = torch.load(vector_path, map_location="cpu", weights_only=True) if vector_path.exists() else None
    if state is not None and (state["pairs"] != all_pairs or state["seed"] != args.seed
                              or state["max_new_tokens"] != args.max_new_tokens):
        p.error("Saved vector settings changed; use another output folder")
    completed = set(state["completed_neutral_ids"]) if state else set()
    pending = [pair for pair in pairs if pair["neutral_id"] not in completed]
    if not pending:
        print("All selected extraction pairs already saved.", flush=True)
        return
    model, tokenizer, feature_extractor = load_model()
    layers = model.llm_model.model.transformer.blocks
    if state is None:
        hidden_size = model.llm_model.get_input_embeddings().weight.shape[1]
        state = {
            "sums": {"angry": torch.zeros(len(layers), args.max_new_tokens, len(GROUPS), hidden_size)},
            "counts": {"angry": torch.zeros(len(layers), args.max_new_tokens, dtype=torch.long)},
            "pair_mean_sums": {"angry": torch.zeros(len(layers), len(GROUPS), hidden_size)},
            "pair_counts": {"angry": torch.zeros(len(layers), dtype=torch.long)},
            "groups": GROUPS, "styles": ["angry"], "max_passes": args.max_new_tokens,
            "prompt": PROMPT, "max_new_tokens": args.max_new_tokens, "steps": 128,
            "classifier": "emotion2vec", "sign": "style minus neutral", "seed": args.seed,
            "pairs": all_pairs, "extraction_question_ids": [r["question_id"] for r in excluded],
            "completed_neutral_ids": [],
        }
    bounds, captured = {}, []
    original_insert = decoder.insert_audio_embedding

    def capture_insert(text_embeds, speech_embeds, audio_info_lengths):
        start = int(audio_info_lengths[0].item())
        bounds.update(start=start, acoustic_end=start+model.config.prompt_size,
                      end=start+speech_embeds.shape[1])
        return original_insert(text_embeds, speech_embeds, audio_info_lengths)

    def make_hook(layer_index):
        def hook(module, inputs, result):
            hidden = result[0]
            slices = [slice(bounds["start"], bounds["acoustic_end"]),
                      slice(bounds["acoustic_end"], bounds["end"]),
                      slice(bounds["start"], bounds["end"]), slice(-args.max_new_tokens, None)]
            captured[layer_index].append(torch.stack([
                hidden[0, positions].float().mean(dim=0) for positions in slices
            ]).cpu())
        return hook

    def capture(clip_id, seed):
        nonlocal captured
        captured = [[] for _ in layers]
        bounds.clear()
        audio, rate = sf.read(manifest[clip_id]["audio_path"], dtype="float32")
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        response = gen(audio, rate, model, tokenizer, feature_extractor, PROMPT, args.max_new_tokens)
        summary = torch.stack([torch.stack(passes) for passes in captured])
        captured = []
        return response, summary

    def save():
        temporary = vector_path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(vector_path)
        print(f'Saved {len(state["completed_neutral_ids"])} pairs: {vector_path}', flush=True)

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    decoder.insert_audio_embedding = capture_insert
    try:
        for index, pair in enumerate(pending, 1):
            print(f'[{index}/{len(pending)}] {pair["question_id"]} neutral + angry', flush=True)
            neutral_response, neutral = capture(pair["neutral_id"], pair["seed"])
            angry_response, angry = capture(pair["angry_id"], pair["seed"])
            n = min(neutral.shape[1], angry.shape[1])
            difference = angry[:, :n] - neutral[:, :n]
            state["sums"]["angry"][:, :n] += difference
            state["counts"]["angry"][:, :n] += 1
            state["pair_mean_sums"]["angry"] += difference.mean(dim=1)
            state["pair_counts"]["angry"] += 1
            state["completed_neutral_ids"].append(pair["neutral_id"])
            with (args.output / "responses.jsonl").open("a") as stream:
                stream.write(json.dumps({**pair, "neutral_response": neutral_response,
                    "angry_response": angry_response, "neutral_forwards": neutral.shape[1],
                    "angry_forwards": angry.shape[1], "matched_forwards": n}) + "\n")
            print(f"  forwards neutral={neutral.shape[1]} angry={angry.shape[1]} matched={n}", flush=True)
            del neutral, angry, difference
            if index % args.save_every == 0:
                save()
    finally:
        for handle in handles:
            handle.remove()
        decoder.insert_audio_embedding = original_insert
        save()


if __name__ == "__main__":
    main()
