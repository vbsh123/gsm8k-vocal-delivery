"""Train additive vectors on first-set answers with DIFFA frozen.

Matched/all describe the supplied initialization. Both use the same masked-answer
loss, not a truncated generation trajectory. Each target is trained independently.
"""

import argparse
import json
from pathlib import Path
import random

from extract_audio_vectors import GROUPS, save_tensor


TARGETS = [(11, "answer", 1.0), (12, "acoustic", 1.0), (16, "answer", 1.25)]


def trainable_hook(vector, positions, strength):
    def hook(module, inputs, output):
        hidden = output[0].clone()
        hidden[:, positions, :] += strength * vector.to(hidden.dtype)
        return (hidden, *output[1:])
    return hook


def masked_answer_loss(logits, target_ids, masked, torch):
    return torch.nn.functional.cross_entropy(logits[0, -512:][masked].float(), target_ids[masked])


def prepare_prefix(recording, model, tokenizer, features, prompt, torch):
    import librosa
    import soundfile as sf
    from src.dataloader import build_chat_prompt
    from src.fast_dllm.diffa_generate import insert_audio_embedding

    audio, rate = sf.read(recording["audio_path"], dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != 16000:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
    with torch.no_grad():
        inputs = features(audio, sampling_rate=16000, truncation=False,
                          return_tensors="pt", return_attention_mask=True).to("cuda")
        speech, _ = model.get_speech_features(inputs.input_features, inputs.attention_mask, stage=model.stage)
        audio_messages = [{"role": "system", "content": ""},
                          {"role": "user", "content": "This is the audio: "}]
        messages = [{"role": "system", "content": ""},
                    {"role": "user", "content": "This is the audio: <audio>. " + prompt}]
        audio_text = build_chat_prompt(audio_messages, add_generation_prompt=False)[:-10]
        text = build_chat_prompt(messages, add_generation_prompt=True)
        audio_tokens = tokenizer(audio_text, return_tensors="pt", max_length=4096, truncation=True)
        text_tokens = tokenizer(text, return_tensors="pt", max_length=4096, truncation=True)
        start = int(audio_tokens.attention_mask.sum().item()) - 1
        embeddings = model.llm_model.get_input_embeddings()(text_tokens.input_ids.to("cuda"))
        prefix = insert_audio_embedding(embeddings, speech, torch.tensor([start], device="cuda"))
    bounds = dict(start=start, acoustic_end=start + model.config.prompt_size, end=start + speech.shape[1])
    return prefix.to(torch.bfloat16).detach(), bounds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--initial-vectors", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--l2", type=float, default=0.01)
    p.add_argument("--limit", type=int)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    collection = json.loads((args.collection / "collection-config.json").read_text())
    pairs = collection["pairs"] if args.limit is None else collection["pairs"][:args.limit]
    records = [collection["manifest"][pair["neutral_id"]] for pair in pairs]
    folder = args.output / "answer_trained"
    folder.mkdir(parents=True, exist_ok=True)
    initial_stat = args.initial_vectors.stat()
    recipe = dict(collection=str(args.collection.resolve()), initial_vectors=str(args.initial_vectors.resolve()),
                  initial_vector_bytes=initial_stat.st_size, initial_vector_mtime_ns=initial_stat.st_mtime_ns,
                  epochs=args.epochs, learning_rate=args.learning_rate, l2=args.l2, seed=42,
                  question_ids=[r["question_id"] for r in records], targets=TARGETS,
                  initialization_averaging=collection.get("averaging", "matched"),
                  objective="Masked correct Final answer tokens on neutral audio; base model frozen")
    recipe = json.loads(json.dumps(recipe))
    path = folder / "training-config.json"
    if path.exists() and json.loads(path.read_text()) != recipe:
        p.error("Training settings changed; use another output")
    path.write_text(json.dumps(recipe, indent=2) + "\n")
    print(f"{len(records)} training questions × {args.epochs} epochs × {len(TARGETS)} independent targets.", flush=True)
    print("Matched/all changes initialization only. No evaluation answers are used in training.", flush=True)
    if args.prepare_only:
        return

    import numpy as np
    import torch
    from extract_vectors import load_model
    from src.fast_dllm.model.configuration_llada import ActivationCheckpointingStrategy

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    base = torch.load(args.initial_vectors, map_location="cpu", weights_only=True)
    if base["prompt"] != collection["prompt"] or base["max_new_tokens"] != 512:
        p.error("Initialization and training must use the same prompt and 512 answer positions")
    excluded = sorted({r["question_id"] for r in collection["manifest"].values()})
    output_path = folder / "style-vectors.pt"
    progress_path = folder / "training-state.pt"
    progress = torch.load(progress_path, map_location="cpu", weights_only=True) if progress_path.exists() else None
    means = base["pair_mean_sums"]["angry"] / base["pair_counts"]["angry"][:, None, None]
    directions = progress["directions"] if progress else means.clone()
    finished = progress["finished"] if progress else []
    if len(finished) == len(TARGETS):
        print("All trained targets already saved:", output_path, flush=True)
        return
    model, tokenizer, features = load_model()
    model.llm_model.model.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    layers = model.llm_model.model.transformer.blocks
    width = means.shape[-1]
    for layer, group, strength in TARGETS:
        name = f"L{layer}-{group}-a{strength:g}"
        if name in finished:
            continue
        vector = torch.nn.Parameter(means[layer, GROUPS.index(group)].clone().to("cuda"))
        optimizer = torch.optim.Adam([vector], lr=args.learning_rate)
        generator = torch.Generator().manual_seed(42)
        cursor = 0
        if progress and progress.get("target") == name:
            vector.data.copy_(progress["vector"].to("cuda"))
            optimizer.load_state_dict(progress["optimizer"])
            generator.set_state(progress["generator"])
            cursor = progress["cursor"]
        order = []
        for epoch in range(args.epochs):
            indices = list(range(len(records)))
            random.Random(42 + epoch).shuffle(indices)
            order.extend(indices)
        for step in range(cursor, len(order)):
            recording = records[order[step]]
            prefix, bounds = prepare_prefix(recording, model, tokenizer, features, collection["prompt"], torch)
            ids = tokenizer.encode("Final answer: " + str(recording["answer"]), add_special_tokens=False)
            ids = ids[:511] + [tokenizer.eos_token_id]
            target = torch.full((512,), 126336, dtype=torch.long, device="cuda")
            target[:len(ids)] = torch.tensor(ids, device="cuda")
            probability = 0.1 + 0.9 * float(torch.rand((), generator=generator))
            masked = torch.rand(len(ids), generator=generator) < probability
            if not masked.any():
                masked[0] = True
            mask = torch.zeros(512, dtype=torch.bool, device="cuda")
            mask[:len(ids)] = masked.to("cuda")
            noisy = target.clone()
            noisy[mask] = 126336
            with torch.no_grad():
                answer_embeddings = model.llm_model.get_input_embeddings()(noisy[None])
                embeddings = torch.cat([prefix, answer_embeddings], dim=1).detach()
            positions = slice(-512, None) if group == "answer" else slice(bounds["start"], bounds["acoustic_end"])
            handle = layers[layer].register_forward_hook(trainable_hook(vector, positions, strength))
            optimizer.zero_grad(set_to_none=True)
            try:
                logits = model.llm_model(inputs_embeds=embeddings, use_cache=False, return_dict=True).logits
                answer_loss = masked_answer_loss(logits, target, mask, torch)
                loss = answer_loss + args.l2 * vector.square().mean()
                loss.backward()
            finally:
                handle.remove()
            if vector.grad is None or not torch.isfinite(vector.grad).all():
                raise RuntimeError("Training vector has no finite gradient")
            torch.nn.utils.clip_grad_norm_([vector], 1.0)
            optimizer.step()
            print(f"{name} [{step+1}/{len(order)}] {recording['question_id']} answer_loss={float(answer_loss.detach()):.4f}", flush=True)
            if (step + 1) % 25 == 0 or step + 1 == len(order):
                save_tensor(torch, dict(directions=directions, finished=finished, target=name,
                                       vector=vector.detach().cpu(), optimizer=optimizer.state_dict(),
                                       generator=generator.get_state(), cursor=step+1), progress_path)
            del logits, loss, answer_loss, embeddings, answer_embeddings, prefix
        directions[layer, GROUPS.index(group)] = vector.detach().cpu()
        finished.append(name)
        state = dict(base)
        state.update(pair_mean_sums={"angry": directions.clone()},
                     pair_counts={"angry": torch.ones(len(layers), dtype=torch.long)},
                     method="answer_trained", sign="learned additive intervention",
                     norm="learned; no post-training rescaling", supported_modes=["fixed"],
                     extraction_question_ids=excluded, trained_targets=list(finished),
                     training_pairs=len(records), training=recipe)
        save_tensor(torch, state, output_path)
        save_tensor(torch, dict(directions=directions, finished=finished), progress_path)
        progress = None
        (folder / "extraction-questions.jsonl").write_text("".join(json.dumps(dict(question_id=q)) + "\n" for q in excluded))
        print("Saved target:", name, flush=True)
    print("Saved:", output_path, flush=True)


if __name__ == "__main__":
    main()
