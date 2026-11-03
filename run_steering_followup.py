"""Run focused strength/combination plans without editing steer.py."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import time

from scoring import score_response
from steer import make_steering_hook, read_rows, reuse_questions, summarize


def build_stages(semantic_layers, plan="original"):
    def condition(name, targets):
        return dict(name=name, interventions=[dict(layer=layer, group=group, strength=strength)
                                             for layer, group, strength in targets])
    if plan == "higher-strength":
        conditions = [condition("baseline", [])]
        for layer, group in [(4, "acoustic"), (18, "answer"), (20, "answer")]:
            conditions.append(condition(f"L{layer}-{group}-a2", [(layer, group, 2)]))
        conditions.append(condition("L4-acoustic+L18-answer-a2", [(4, "acoustic", 2), (18, "answer", 2)]))
        conditions.append(condition("L18-answer+L20-answer-a1", [(18, "answer", 1), (20, "answer", 1)]))
        return [dict(name="higher-strength", conditions=conditions)]
    first = [condition("baseline", [])]
    for layer, group in [(4, "acoustic"), (2, "answer"), (18, "answer"), (20, "answer")]:
        first.append(condition(f"L{layer}-{group}-a0.5", [(layer, group, 0.5)]))
    first.append(condition("L4-acoustic+L18-answer-a1", [(4, "acoustic", 1), (18, "answer", 1)]))
    first.append(condition("L18-answer+L20-answer-a0.5", [(18, "answer", 0.5), (20, "answer", 0.5)]))
    semantic = [condition(f"L{layer}-semantic-a1", [(layer, "semantic", 1)])
                for layer in semantic_layers]
    if any(layer < 0 for layer in semantic_layers) or len(set(semantic_layers)) != len(semantic_layers):
        raise ValueError("Semantic layer indices must be nonnegative and unique")
    return [dict(name="strength-and-combinations", conditions=first),
            dict(name="semantic", conditions=semantic)]


@contextmanager
def install_interventions(layers, directions, interventions, bounds, answer_tokens):
    """Keep all hooks for a combination active during the SAME generation."""
    handles, states = [], []
    targets = interventions or [dict(layer=0, group="answer", strength=0)]
    try:
        for target in targets:
            vector = directions[target["layer"], target["group"]] if target["strength"] else None
            hook, state = make_steering_hook(vector, None, "fixed", target["group"],
                                             target["strength"], bounds, answer_tokens)
            handles.append(layers[target["layer"]].register_forward_hook(hook))
            states.append(dict(target=target, state=state))
        yield states
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", choices=["original", "higher-strength"], default="original",
                        help="higher-strength runs five stronger interventions plus baseline, with no semantic sweep")
    parser.add_argument("--path", type=Path, default=Path("outputs/manual200-mixed/cropped"))
    parser.add_argument("--vectors", type=Path, default=Path("outputs/manual200-mixed/style_vectors.pt"))
    parser.add_argument("--questions-from", type=Path,
                        default=Path("outputs/steer-angry-L15-25-dev200-a1/run_config.json"))
    parser.add_argument("--exclude-results", type=Path, nargs="+", default=[
        Path("outputs/manual200-mixed/vector-extraction-questions.jsonl"),
        Path("outputs/steer-angry-all32-dev/results.jsonl")])
    parser.add_argument("--num-examples", type=int, default=200,
                        help="First N recordings from the saved question plan; use 1 for a GPU smoke")
    parser.add_argument("--semantic-layers", type=int, nargs="+", default=list(range(26)),
                        help="Semantic positions at these layers, tested separately AFTER the first stage")
    parser.add_argument("--output", type=Path, default=Path("outputs/steer-angry-followup200"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.num_examples <= 0:
        parser.error("num-examples must be positive")
    if not all(p.is_file() for p in args.exclude_results):
        parser.error("Exclusion files must exist")
    stages = build_stages(args.semantic_layers, args.plan)
    conditions = [c for stage in stages for c in stage["conditions"]]
    previous = json.loads(args.questions_from.read_text())["recordings"]
    if args.num_examples > len(previous):
        parser.error(f"Previous plan has only {len(previous)} questions")
    records = {r["clip_id"]: r for r in read_rows(args.path / "manifest.jsonl")}
    excluded = {r["question_id"] for path in args.exclude_results for r in read_rows(path)}
    selected = reuse_questions(previous[:args.num_examples], records, excluded, args.num_examples)
    stat = args.vectors.stat()
    settings = dict(path=str(args.path.resolve()), vectors=str(args.vectors.resolve()),
                    vector_bytes=stat.st_size, vector_mtime_ns=stat.st_mtime_ns,
                    questions_from=str(args.questions_from.resolve()),
                    exclude_results=[str(p.resolve()) for p in args.exclude_results],
                    seed=args.seed, style="angry", mode="fixed", stages=stages, conditions=conditions)
    plan = dict(settings=settings, recordings=selected, excluded_question_ids=sorted(excluded))
    config_path = args.output / "run_config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved["settings"] != settings or saved["recordings"] != selected:
            parser.error("Follow-up settings/recordings changed; use a fresh output folder")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(plan, indent=2) + "\n")
    results_path = args.output / "results.jsonl"
    results = read_rows(results_path)
    completed = {(r["clip_id"], r["condition"]) for r in results}
    total = len(selected) * len(conditions)
    pending = sum((r["clip_id"], c["name"]) not in completed for r in selected for c in conditions)
    print(f"{len(selected)} questions, {len(conditions)} conditions, {total} generations; {pending} remain.", flush=True)
    for stage in stages:
        print(f"{stage['name']}: {len(stage['conditions']) * len(selected)} generations", flush=True)
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
    if data.get("sign") != "style minus neutral":
        raise ValueError("Unknown vector sign convention")
    prompt, answer_tokens = data["prompt"], data["max_new_tokens"]
    targets = {(t["layer"], t["group"]) for c in conditions for t in c["interventions"]}
    directions = {}
    for layer, group in sorted(targets):
        if layer >= data["pair_mean_sums"]["angry"].shape[0]:
            raise ValueError(f"Invalid layer index: {layer}")
        group_index = data["groups"].index(group)
        count = int(data["pair_counts"]["angry"][layer])
        if count == 0:
            raise ValueError(f"No angry extraction pairs at layer {layer}")
        vector = data["pair_mean_sums"]["angry"][layer, group_index] / count
        if not torch.isfinite(vector).all():
            raise ValueError("Direction contains non-finite values")
        directions[layer, group] = vector
    model, tokenizer, feature_extractor = load_model()
    layers = model.llm_model.model.transformer.blocks
    directions = {key: value.to(device="cuda", dtype=torch.bfloat16) for key, value in directions.items()}
    bounds = {}
    original_insert = decoder.insert_audio_embedding

    def capture_insert(text_embeds, speech_embeds, audio_info_lengths):
        start = int(audio_info_lengths[0].item())
        bounds.update(start=start, acoustic_end=start+model.config.prompt_size,
                      end=start+speech_embeds.shape[1])
        return original_insert(text_embeds, speech_embeds, audio_info_lengths)

    decoder.insert_audio_embedding = capture_insert
    try:
        for stage in stages:
            print(f"\nStarting stage: {stage['name']}", flush=True)
            for index, r in enumerate(selected, 1):
                if all((r["clip_id"], c["name"]) in completed for c in stage["conditions"]):
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
                for c_index, c in enumerate(stage["conditions"], 1):
                    if (r["clip_id"], c["name"]) in completed:
                        continue
                    seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{r['clip_id']}".encode()).digest()[:4], "big")
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    bounds.clear()
                    print(f"[{stage['name']}] [{index}/{len(selected)}] "
                          f"[{c_index}/{len(stage['conditions'])}] {r['clip_id']} {c['name']}", flush=True)
                    started = time.monotonic()
                    with install_interventions(layers, directions, c["interventions"], bounds, answer_tokens) as states:
                        with torch.inference_mode():
                            response = model.generate(
                                tokenizer, input_audio_features=inputs.input_features,
                                input_audio_mask=inputs.attention_mask, system_prompt="",
                                input_texts=[prompt], steps=128, block_length=32,
                                max_new_tokens=answer_tokens, temperature=0.0, accelerate="fast_dllm")
                    diagnostics = [dict(**s["target"], **s["state"]) for s in states]
                    result = dict(clip_id=r["clip_id"], question_id=r["question_id"], stage=stage["name"],
                                  condition=c["name"], interventions=c["interventions"],
                                  vector_style="angry", mode="fixed", seed=seed, response=response,
                                  generation_seconds=time.monotonic()-started,
                                  prompt=prompt, max_new_tokens=answer_tokens, decoder="fast_dllm",
                                  forwards=diagnostics[0]["forwards"],
                                  steered_forwards=max(s["steered"] for s in diagnostics),
                                  intervention_forwards=diagnostics,
                                  **score_response(response, r["answer"]))
                    with results_path.open("a") as f:
                        f.write(json.dumps(result) + "\n")
                    results.append(result)
                    completed.add((r["clip_id"], c["name"]))
                    print(f"  correct={result['correct']} answer={result['predicted_answer']} "
                          f"intervention_forwards={diagnostics}", flush=True)
            summary = summarize(results, conditions)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            (args.output / f"summary-{stage['name']}.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        decoder.insert_audio_embedding = original_insert


if __name__ == "__main__":
    main()
