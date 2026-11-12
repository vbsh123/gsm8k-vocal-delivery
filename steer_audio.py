"""Steer recordings using an explicit JSON list of recordings and conditions."""

import argparse
import json
from pathlib import Path
import random
import time

from steer import install_interventions, diffusion_progress, direction_settings
from scoring import score_response
from steer import read_rows, summarize


def run_experiment(recordings, conditions, vectors, output, audio_root=Path("."),
                   allow_extraction_overlap=False, report=None, summary_name="summary.json"):
    if report is None:
        report = lambda rows, conditions, recordings: summarize(rows, conditions)
    results_path = output / "results.jsonl"
    results = read_rows(results_path, repair=True)
    if any(row.get("seed") != 42 for row in results):
        raise ValueError("Cached answers use a different seed; use a new output folder")
    completed = {(r["clip_id"], r["condition"]) for r in results}
    if all((r["clip_id"], c["name"]) in completed for r in recordings for c in conditions):
        return report(results, conditions, recordings)

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    import src.fast_dllm.diffa_generate as decoder
    from extract_vectors import load_model

    data = torch.load(vectors, map_location="cpu", weights_only=True, mmap=True)
    selected_ids = {r["question_id"] for r in recordings}
    if set(data.get("extraction_question_ids", [])) & selected_ids and not allow_extraction_overlap:
        raise ValueError("Evaluation questions overlap extraction metadata")
    prompt, answer_tokens = data["prompt"], data["max_new_tokens"]
    mode = data.get("mode", "fixed")
    window_size = data.get("window_size", 20)
    directions = {}
    for layer, group in {(t["layer"], t["group"]) for c in conditions for t in c["interventions"]}:
        count = int(data["pair_counts"]["angry"][layer])
        if count == 0:
            raise ValueError(f'No direction was collected for layer {layer}')
        if "collected_targets" in data and (layer, group) not in map(tuple, data["collected_targets"]):
            raise ValueError(f'No direction was collected for L{layer} {group}')
        vector = data["pair_mean_sums"]["angry"][layer, data["groups"].index(group)] / count
        if mode in ("phase", "forward_windows"):
            counts = data["temporal_counts"]["angry"][layer]
            sums = data["temporal_sums"]["angry"][layer, :, data["groups"].index(group)]
            bins = sums / counts.clamp_min(1)[:, None]
            bins[counts == 0] = vector
            vector = torch.cat([bins, vector[None]]) if mode == "forward_windows" else bins
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
        for index, recording in enumerate(recordings, 1):
            if all((recording["clip_id"], c["name"]) in completed for c in conditions):
                continue
            audio_path = Path(recording["audio_path"])
            if not audio_path.is_absolute():
                audio_path = audio_root / audio_path
            audio, rate = sf.read(audio_path, dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if rate != 16000:
                audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
            inputs = feature_extractor(audio, sampling_rate=16000, truncation=False,
                                       return_tensors="pt", return_attention_mask=True).to("cuda")
            for c_index, c in enumerate(conditions, 1):
                if (recording["clip_id"], c["name"]) in completed:
                    continue
                seed = 42
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                bounds.clear()
                print(f'[{index}/{len(recordings)}] [{c_index}/{len(conditions)}] '
                      f'{recording["question_id"]} {c["name"]}', flush=True)
                started = time.monotonic()
                schedule = c.get("schedule", "always")
                timed = mode != "fixed" or schedule != "always"
                with diffusion_progress(model, answer_tokens, enabled=timed) as progress, install_interventions(
                        layers, directions, c["interventions"], bounds, answer_tokens,
                        mode, progress, schedule, window_size) as states:
                    with torch.inference_mode():
                        response = model.generate(
                            tokenizer, input_audio_features=inputs.input_features,
                            input_audio_mask=inputs.attention_mask, system_prompt="",
                            input_texts=[prompt], steps=128, block_length=32,
                            max_new_tokens=answer_tokens, temperature=0.0, accelerate="fast_dllm")
                diagnostics = [dict(**s["target"], **s["state"]) for s in states]
                row = {"clip_id": recording["clip_id"], "question_id": recording["question_id"],
                       "condition": c["name"], "interventions": c["interventions"],
                       "vector_style": "angry", "mode": mode, "seed": seed,
                       "response": response, "prompt": prompt, "max_new_tokens": answer_tokens,
                       "decoder": "fast_dllm", "generation_seconds": time.monotonic()-started,
                       "forwards": diagnostics[0]["forwards"],
                       "steered_forwards": max(s["steered"] for s in diagnostics),
                       "intervention_forwards": diagnostics,
                       **score_response(response, recording["answer"])}
                if timed:
                    row.update(schedule=schedule, phase_forwards=progress["phase_counts"])
                with results_path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                results.append(row)
                completed.add((recording["clip_id"], c["name"]))
                print(f'  correct={row["correct"]} answer={row["predicted_answer"]}', flush=True)
    finally:
        decoder.insert_audio_embedding = original_insert
    summary = report(results, conditions, recordings)
    (output / summary_name).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def select_conditions(conditions, names):
    if not names:
        return conditions
    unknown = set(names) - {c['name'] for c in conditions}
    if unknown:
        raise ValueError(f'Unknown conditions: {sorted(unknown)}')
    return [c for c in conditions if c['name']=='baseline' or c['name'] in names]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, help="JSON with recordings and conditions; saved run_config.json also works.")
    p.add_argument("--config", type=Path, help="Run an already prepared run_config.json.")
    p.add_argument("--vectors", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audio-root", type=Path, default=Path("."))
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--conditions", nargs='+', help="Resume only these conditions from an existing --config, retaining baseline.")
    p.add_argument("--allow-extraction-overlap", action="store_true")
    args = p.parse_args(argv)
    config_path = args.output / "run_config.json"
    if args.summary_only:
        saved = json.loads(config_path.read_text())
        return summarize(read_rows(args.output / "results.jsonl"), select_conditions(saved["settings"]["conditions"], args.conditions))
    if args.config:
        saved = json.loads(args.config.read_text())
        settings = saved["settings"]
        selected = select_conditions(settings['conditions'], args.conditions)
        if settings.get("seed_policy") != "constant_before_every_generation" or settings["seed"] != 42:
            p.error("This runner requires the seed42 generation recipe")
        vectors = Path(settings["vectors"])
        stat = vectors.stat()
        if (stat.st_size, stat.st_mtime_ns) != (settings["vector_bytes"], settings["vector_mtime_ns"]):
            p.error("Vector checkpoint changed; use a new run")
        if not args.prepare_only:
            return run_experiment(saved["recordings"], selected, vectors, args.output,
                                  Path(settings.get("path", ".")), settings.get("allow_extraction_overlap", False),
                                  summary_name='summary-selected.json' if args.conditions else 'summary.json')
        return
    if args.conditions:
        p.error('--conditions requires an existing --config or --summary-only')
    if args.plan is None or args.vectors is None:
        p.error("--plan and --vectors are required for a run")
    source = json.loads(args.plan.read_text())
    recordings = source["recordings"]
    conditions = source["conditions"] if "conditions" in source else source["settings"]["conditions"]
    excluded = {r["question_id"] for r in read_rows(args.vectors.parent / "extraction-questions.jsonl")}
    if excluded & {r["question_id"] for r in recordings} and not args.allow_extraction_overlap:
        p.error("Evaluation questions overlap vector extraction")
    stat = args.vectors.stat()
    settings = dict(vectors=str(args.vectors.resolve()), vector_bytes=stat.st_size,
                    vector_mtime_ns=stat.st_mtime_ns, seed=42,
                    seed_policy="constant_before_every_generation", style="angry", mode="fixed",
                    path=str(args.audio_root.resolve()), conditions=conditions,
                    allow_extraction_overlap=args.allow_extraction_overlap)
    settings.update(direction_settings(args.vectors))
    config = dict(settings=settings, recordings=recordings)
    args.output.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        p.error("Saved run changed; use a new output folder")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    completed = {(r["clip_id"], r["condition"]) for r in read_rows(args.output / "results.jsonl")}
    pending = sum((r["clip_id"], c["name"]) not in completed for r in recordings for c in conditions)
    print(f"{len(recordings)} recordings × {len(conditions)} conditions; {pending} generations pending; seed42.", flush=True)
    if not args.prepare_only:
        return run_experiment(recordings, conditions, args.vectors, args.output, args.audio_root,
                              args.allow_extraction_overlap)


if __name__ == "__main__":
    main()
