"""Outcome-selected six-question diagnostic of angry steering strengths and combinations."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

from run_steering_followup import install_interventions
from scoring import score_response
from steer import read_rows, summarize


QUESTIONS = ["gsm8k_test_00708", "gsm8k_test_00736", "gsm8k_test_00746",
             "gsm8k_test_00781", "gsm8k_test_00792", "gsm8k_test_00987"]


def build_conditions(config, results, questions, plan="original"):
    latest = {(r["question_id"], r["condition"]): r for r in results}

    def correct(row):
        return score_response(row["response"], row["reference_answer"])["correct"]

    for qid in questions:
        baseline = latest.get((qid, "baseline"))
        if baseline is None or correct(baseline):
            raise ValueError(f"Expected a saved incorrect baseline for {qid}")
    recovered = set()
    for c in config["settings"]["conditions"]:
        if c["name"] == "baseline":
            continue
        if any((qid, c["name"]) in latest and correct(latest[qid, c["name"]]) for qid in questions):
            recovered.add((c["layer"], c["group"]))

    def condition(name, targets):
        return {"name": name, "interventions": [
            {"layer": layer, "group": group, "strength": strength} for layer, group, strength in targets]}

    if plan == "lower-strength":
        conditions = [condition("baseline", []),
                      condition("L5ac1+L11ans1+L3ans2", [
                          (5, "acoustic", 1), (11, "answer", 1), (3, "answer", 2)])]
        excluded = {(5, "acoustic"), (11, "answer"), (3, "answer")}
        for layer, group in sorted(recovered - excluded):
            conditions.append(condition(f"L{layer}-{group}-a0.5", [(layer, group, 0.5)]))
        return conditions

    conditions = [condition("baseline", []), condition("L5-acoustic-a2", [(5, "acoustic", 2)])]
    for layer, group, short in [(0, "acoustic", "L0-ac"), (11, "answer", "L11-ans")]:
        for strength in (1, 0.5):
            conditions.append(condition(f"L5-ac1+{short}{strength:g}",
                              [(5, "acoustic", 1), (layer, group, strength)]))
    for layer, group in sorted(recovered - {(5, "acoustic")}):
        conditions.append(condition(f"L{layer}-{group}-a2", [(layer, group, 2)]))
    return conditions


def report(results, conditions, recordings):
    summary = summarize(results, conditions)
    latest = {(r["question_id"], r["condition"]): r for r in results}
    print("\nPer-question correct conditions (outcome-selected diagnostic):")
    for recording in recordings:
        qid = recording["question_id"]
        baseline = latest.get((qid, "baseline"))
        fixed = [c["name"] for c in conditions[1:] if latest.get((qid, c["name"]), {}).get("correct")]
        print(f'{qid}: fresh baseline={baseline["correct"] if baseline else "pending"}; '
              f'correct conditions: {", ".join(fixed) or "none yet"}')
    print("Outcome-selected diagnostic: these accuracies/p-values do not estimate general steering benefit.")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, default=Path("outputs/steer-jenny-angry-L0-25-n50"))
    p.add_argument("--vectors", type=Path, default=Path("outputs/jenny-angry-vectors/style-vectors.pt"))
    p.add_argument("--plan", choices=["original", "lower-strength"], default="original")
    p.add_argument("--prepared-plan", type=Path, help="Explicit recordings and intervention conditions from a diagnostic adapter.")
    p.add_argument("--output", type=Path)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--fixed-seed", type=int, help="Reset exactly this seed before every generation; default preserves per-clip derived seeds.")
    p.add_argument("--allow-extraction-overlap", action="store_true",
                   help="Explicit in-sample diagnostic only; otherwise extraction overlap is rejected.")
    args = p.parse_args(argv)
    if args.fixed_seed is not None and not 0 <= args.fixed_seed < 2**32:
        p.error("--fixed-seed must be in 0..2**32-1")
    args.output = args.output or Path("outputs/steer-jenny-angry-six-" + (
        "half" if args.plan == "lower-strength" else "followup"))
    config_path = args.output / "run_config.json"
    results_path = args.output / "results.jsonl"
    if args.summary_only:
        saved = json.loads(config_path.read_text())
        report(read_rows(results_path), saved["settings"]["conditions"], saved["recordings"])
        return
    previous = json.loads((args.previous / "run_config.json").read_text())
    if args.prepared_plan:
        prepared = json.loads(args.prepared_plan.read_text())
        selected, conditions = prepared["recordings"], prepared["conditions"]
        if not selected or conditions[0]["name"] != "baseline" or conditions[0]["interventions"]:
            p.error("Prepared diagnostic needs recordings and an unsteered first baseline")
        if len({r["clip_id"] for r in selected}) != len(selected):
            p.error("Duplicate recordings in prepared plan")
        if len({c["name"] for c in conditions}) != len(conditions):
            p.error("Duplicate conditions in prepared plan")
    else:
        records = {r["question_id"]: r for r in previous["recordings"]}
        selected = [records[qid] for qid in QUESTIONS]
        conditions = build_conditions(previous, read_rows(args.previous / "results.jsonl"), QUESTIONS, args.plan)
    selected_ids = {r["question_id"] for r in selected}
    excluded = {r["question_id"] for r in read_rows(args.vectors.parent / "extraction-questions.jsonl")}
    if excluded & selected_ids and not args.allow_extraction_overlap:
        p.error("The diagnostic overlaps vector-extraction questions")
    stat = args.vectors.stat()
    settings = {"previous": str(args.previous.resolve()), "vectors": str(args.vectors.resolve()),
                "vector_bytes": stat.st_size, "vector_mtime_ns": stat.st_mtime_ns,
                "seed": previous["settings"]["seed"], "conditions": conditions,
                "style": "angry", "mode": "fixed", "selection": "outcome-selected diagnostic"}
    if args.fixed_seed is not None:
        settings.update(seed=args.fixed_seed, seed_policy="constant_before_every_generation")
    if args.plan != "original":
        settings["plan"] = args.plan
    if args.allow_extraction_overlap:
        settings["allow_extraction_overlap"] = True
        print("IN-SAMPLE diagnostic: extraction-question overlap is explicitly allowed.", flush=True)
    if args.prepared_plan:
        settings.update(prepared_plan=str(args.prepared_plan.resolve()),
                        prepared_plan_sha256=hashlib.sha256(args.prepared_plan.read_bytes()).hexdigest())
    plan = {"settings": settings, "recordings": selected}
    args.output.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and json.loads(config_path.read_text()) != plan:
        p.error("Saved diagnostic settings changed; use a new --output")
    config_path.write_text(json.dumps(plan, indent=2) + "\n")
    results = read_rows(results_path)
    completed = {(r["clip_id"], r["condition"]) for r in results}
    pending = sum((r["clip_id"], c["name"]) not in completed for r in selected for c in conditions)
    print(f"{len(selected)} questions x {len(conditions)} conditions = "
          f"{len(selected)*len(conditions)} generations; {pending} pending.", flush=True)
    if args.prepared_plan:
        print("Explicit diagnostic targets; each condition runs separately against a fresh baseline.", flush=True)
    elif args.plan == "lower-strength":
        print("Combination: L5 acoustic x1 + L11 answer x1 + L3 answer x2. Other targets individually x0.5.", flush=True)
    else:
        print("Combinations: L5 acoustic + L0 acoustic, or L5 acoustic + L11 answer.", flush=True)
    for c in conditions:
        print(f'  {c["name"]}: {c["interventions"]}', flush=True)
    if args.prepare_only:
        return
    if not pending:
        report(results, conditions, selected)
        return

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    import src.fast_dllm.diffa_generate as decoder
    from extract_vectors import load_model

    data = torch.load(args.vectors, map_location="cpu", weights_only=True, mmap=True)
    if data.get("sign") != "style minus neutral":
        raise ValueError("Expected angry-minus-neutral vectors")
    if set(data.get("extraction_question_ids", [])) & selected_ids and not args.allow_extraction_overlap:
        raise ValueError("Diagnostic questions overlap extraction metadata")
    prompt, answer_tokens = data["prompt"], data["max_new_tokens"]
    directions = {}
    for layer, group in {(t["layer"], t["group"]) for c in conditions for t in c["interventions"]}:
        count = int(data["pair_counts"]["angry"][layer])
        if count == 0:
            raise ValueError(f"No angry extraction pairs at layer {layer}")
        vector = data["pair_mean_sums"]["angry"][layer, data["groups"].index(group)] / count
        if not torch.isfinite(vector).all():
            raise ValueError("Non-finite steering direction")
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
        for index, recording in enumerate(selected, 1):
            if all((recording["clip_id"], c["name"]) in completed for c in conditions):
                continue
            audio_path = Path(recording["audio_path"])
            if not audio_path.is_absolute():
                audio_path = Path(previous["settings"]["path"]) / audio_path
            audio, rate = sf.read(audio_path, dtype="float32")
            if len(audio) > 30 * rate:
                raise ValueError(f"Audio over 30 seconds: {audio_path}")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if rate != 16000:
                audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
            inputs = feature_extractor(audio, sampling_rate=16000, truncation=False,
                                       return_tensors="pt", return_attention_mask=True).to("cuda")
            for c_index, c in enumerate(conditions, 1):
                if (recording["clip_id"], c["name"]) in completed:
                    continue
                seed = args.fixed_seed if args.fixed_seed is not None else int.from_bytes(hashlib.sha256(
                    f'{settings["seed"]}:{recording["clip_id"]}'.encode()).digest()[:4], "big")
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                bounds.clear()
                print(f'[{index}/{len(selected)}] [{c_index}/{len(conditions)}] '
                      f'{recording["question_id"]} {c["name"]}', flush=True)
                started = time.monotonic()
                with install_interventions(layers, directions, c["interventions"], bounds, answer_tokens) as states:
                    with torch.inference_mode():
                        response = model.generate(
                            tokenizer, input_audio_features=inputs.input_features,
                            input_audio_mask=inputs.attention_mask, system_prompt="",
                            input_texts=[prompt], steps=128, block_length=32,
                            max_new_tokens=answer_tokens, temperature=0.0, accelerate="fast_dllm")
                diagnostics = [dict(**s["target"], **s["state"]) for s in states]
                row = {"clip_id": recording["clip_id"], "question_id": recording["question_id"],
                       "condition": c["name"], "interventions": c["interventions"],
                       "vector_style": "angry", "mode": "fixed", "seed": seed,
                       "response": response, "prompt": prompt, "max_new_tokens": answer_tokens,
                       "decoder": "fast_dllm", "generation_seconds": time.monotonic()-started,
                       "forwards": diagnostics[0]["forwards"],
                       "steered_forwards": max(s["steered"] for s in diagnostics),
                       "intervention_forwards": diagnostics,
                       **score_response(response, recording["answer"])}
                with results_path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                results.append(row)
                completed.add((recording["clip_id"], c["name"]))
                print(f'  correct={row["correct"]} answer={row["predicted_answer"]}', flush=True)
    finally:
        decoder.insert_audio_embedding = original_insert
    summary = report(results, conditions, selected)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
