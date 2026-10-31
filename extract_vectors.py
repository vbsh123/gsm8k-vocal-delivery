"""Extract per-layer delivery directions from DIFFA's first denoising pass."""

import argparse
import hashlib
import json
from pathlib import Path


SOLVE_PROMPT = ("Solve the math problem in the audio. Give concise step-by-step reasoning "
              "with only the necessary calculations. If the question asks for a total, "
              "compute that total. End with one plain-text line in exactly this format: "
              "Final answer: <number>. Put a single numeric answer on that line, "
              "without Markdown, units, or separate component counts.")


def load_model():
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperModel
    from peft import LoraConfig, get_peft_model
    from src.configuration_DIFFA import DIFFAConfig
    from src.modeling_DIFFA import DIFFAModel
    from src.fast_dllm.model.modeling_llada import LLaDAModelLM

    llm_id = "GSAI-ML/LLaDA-8B-Instruct"
    whisper_id = "openai/whisper-large-v3"
    diffa_id = "zhoujiaming777/DIFFA-2"

    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_id)
    config_path = hf_hub_download(diffa_id, "config.json")
    raw_config = json.loads(Path(config_path).read_text())
    config = DIFFAConfig(**raw_config)
    config.whisper_model_id = whisper_id
    config.prompt_size = raw_config.get("prompt_size", 64)

    model = DIFFAModel(config, tokenizer=tokenizer, stage=2)
    weights_path = hf_hub_download(diffa_id, 'pytorch_model.bin')
    state = torch.load(
        weights_path, map_location="cpu", weights_only=True, mmap=True
    )
    audio_weights = {
        key: value for key, value in state.items() if not key.startswith("llm_model.")
    }
    model.load_state_dict(audio_weights, strict=False)
    whisper = WhisperModel.from_pretrained(whisper_id, torch_dtype= torch.float32)
    model.whisper_model.load_state_dict(whisper.encoder.state_dict())

    del whisper, audio_weights
    llm = LLaDAModelLM.from_pretrained(llm_id, torch_dtype=torch.bfloat16)
    llm_weights = {
        key.removeprefix("llm_model."): value
        for key, value in state.items()
        if key.startswith("llm_model.")
    }
    lora_weights = {
        key: value
        for key, value in llm_weights.items()
        if "lora_" in key
    }

    rank = next(
        value.shape[0]
        for key, value in lora_weights.items()
        if ".lora_A." in key
    )
    targets = sorted({
        key.split(".lora_")[0].split(".")[-1]
        for key in lora_weights
    })

    llm = get_peft_model(
        llm,
        LoraConfig(
            r=rank,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    peft_weights = {
        key if key.startswith("base_model.model.")
        else "base_model.model." + key: value
        for key, value in llm_weights.items()
    }
    llm.load_state_dict(peft_weights, strict=False)

    model.llm_model = llm.merge_and_unload()
    model = model.cuda().eval()
    model.requires_grad_(False)

    return model, tokenizer, feature_extractor


def read_latest(path):
    return {r["clip_id"]: r for line in path.read_text().splitlines()
            if line.strip() for r in [json.loads(line)]}


def question_split(question_id, seed):
    value = int(hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()[:8], 16) / 2**32
    return "train" if value < 0.7 else "validation" if value < 0.85 else "test"


def make_pairs(records, styles, split, seed, num_questions):
    # Pair the same question, voice and repetition; never use accuracy to select pairs.
    neutrals = {(r["question_id"], r["speaker"], r["repeat"]): r
                for r in records.values() if r["style"] == "neutral"}
    pairs = []
    for r in records.values():
        if r["style"] not in styles or question_split(r["question_id"], seed) != split:
            continue
        neutral = neutrals.get((r["question_id"], r["speaker"], r["repeat"]))
        if neutral and max(r["duration_seconds"], neutral["duration_seconds"]) <= 30:
            pairs.append({"question_id": r["question_id"], "style": r["style"],
                          "style_clip_id": r["clip_id"], "neutral_clip_id": neutral["clip_id"]})
    if num_questions:
        selected = set(sorted({p["question_id"] for p in pairs})[:num_questions])
        pairs = [p for p in pairs if p["question_id"] in selected]
    return sorted(pairs, key=lambda p: (p["question_id"], p["style"], p["style_clip_id"]))


class FirstPassComplete(Exception):
    """Stop once the last decoder block has been measured, before logits/decoding."""


def capture_first_pass(model, tokenizer, inputs, prompt, answer_tokens):
    import torch

    transformer = model.llm_model.model.transformer
    blocks = (list(transformer.blocks) if "blocks" in transformer
              else [block for group in transformer.block_groups for block in group])
    captured = {}

    def make_hook(index):
        def hook(module, args, output):
            hidden = output[0]
            # The final answer_tokens positions are all MASKs on the first pass.
            captured[index] = hidden[0, -answer_tokens:].float().mean(dim=0).cpu()
            if index == len(blocks) - 1:
                raise FirstPassComplete
        return hook

    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(blocks)]
    try:
        with torch.inference_mode():
            model.generate(
                tokenizer, input_audio_features=inputs.input_features,
                input_audio_mask=inputs.attention_mask, system_prompt="",
                input_texts=[prompt], steps=128, block_length=32,
                max_new_tokens=answer_tokens, temperature=0.0, accelerate="fast_dllm",
            )
    except FirstPassComplete:
        pass
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(blocks) or not captured:
        raise RuntimeError("Did not capture every LLaDA decoder block.")
    return torch.stack([captured[i] for i in range(len(blocks))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True, help="Cropped recordings folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--styles", nargs="+", default=["angry", "happy", "confident"],
                        choices=["angry", "happy", "confident", "hesitant", "sad", "stressed"])
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-questions", type=int, default=0, help="0 means all eligible questions in this split")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true", help="Reuse saved activations; no GPU/model needed")
    parser.add_argument("--filter-results", type=Path, help="Only average style clips classified NO (emotional)")
    parser.add_argument("--require-neutral", action="store_true", help="Also require neutral classified YES")
    parser.add_argument("--vectors-file", default="vectors.pt")
    args = parser.parse_args()
    if args.num_questions < 0:
        parser.error("num-questions must be nonnegative")
    if args.require_neutral and not args.filter_results:
        parser.error("require-neutral needs filter-results")

    records = read_latest(args.path / "manifest.jsonl")
    pairs = make_pairs(records, args.styles, args.split, args.seed, args.num_questions)
    if not pairs:
        parser.error("No eligible pairs in the selected split")
    clip_ids = sorted({p[key] for p in pairs for key in ("style_clip_id", "neutral_clip_id")})
    # Freeze this extraction's inputs. Use a new output folder for a later dataset snapshot.
    config = {"split": args.split, "seed": args.seed, "styles": args.styles,
              "path": str(args.path.resolve()), "answer_tokens": 512,
              "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "measurement": "first_pass_block_output_mean_over_answer_masks",
              "pairs": pairs, "audio_sha256": {cid: records[cid]["audio_sha256"] for cid in clip_ids}}
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "extraction_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        parser.error("Extraction inputs changed; use a new output folder")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    cache = args.output / "activations"
    cache.mkdir(exist_ok=True)
    pending = [cid for cid in clip_ids if not (cache / f"{cid}.pt").exists()]
    print(f"{len(pairs)} pairs, {len(clip_ids)} clips, {len(pending)} activations remaining; split={args.split}", flush=True)
    if args.prepare_only:
        return
    if args.aggregate_only and pending:
        parser.error("Some activations are missing; run extraction first")

    if pending:
        import librosa
        import soundfile as sf
        import torch

        model, tokenizer, feature_extractor = load_model()
        for index, cid in enumerate(pending, 1):
            r = records[cid]
            audio, rate = sf.read(args.path / r["audio_path"], dtype="float32")
            if len(audio) > 30 * rate:
                raise ValueError(f"{cid} exceeds 30 seconds in the actual WAV")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if rate != 16000:
                audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
            inputs = feature_extractor(audio, sampling_rate=16000, truncation=False,
                                       return_tensors="pt", return_attention_mask=True).to("cuda")
            print(f"[{index}/{len(pending)}] extract {cid}", flush=True)
            hidden = capture_first_pass(model, tokenizer, inputs, SOLVE_PROMPT, config["answer_tokens"])
            target = cache / f"{cid}.pt"
            temporary = target.with_suffix(".tmp.pt")
            torch.save({"clip_id": cid, "hidden": hidden, "prompt": SOLVE_PROMPT}, temporary)
            temporary.replace(target)
        del model
        torch.cuda.empty_cache()

    import torch

    classifications = read_latest(args.filter_results) if args.filter_results else None
    directions, unit_directions, norms, used = {}, {}, {}, {}
    for style in args.styles:
        selected = [p for p in pairs if p["style"] == style]
        if classifications is not None:
            selected = [p for p in selected
                        if classifications.get(p["style_clip_id"], {}).get("predicted_style") == "no"
                        and (not args.require_neutral or
                             classifications.get(p["neutral_clip_id"], {}).get("predicted_style") == "yes")]
        if not selected:
            print(f"{style}: no passing pairs; no vector saved for this style", flush=True)
            continue
        total = None
        for pair in selected:
            styled = torch.load(cache / f"{pair['style_clip_id']}.pt", map_location="cpu", weights_only=True)["hidden"]
            neutral = torch.load(cache / f"{pair['neutral_clip_id']}.pt", map_location="cpu", weights_only=True)["hidden"]
            difference = styled - neutral
            total = difference if total is None else total + difference
        vector = total / len(selected)
        directions[style] = vector
        norms[style] = vector.norm(dim=-1)
        unit_directions[style] = vector / norms[style].clamp_min(1e-12).unsqueeze(-1)
        used[style] = selected
        print(f"{style}: {len(selected)} pairs, vector shape {tuple(vector.shape)}", flush=True)
    target = args.output / args.vectors_file
    temporary = target.with_suffix(".tmp.pt")
    torch.save({"directions": directions, "unit_directions": unit_directions, "norms": norms,
                "pairs": used, "config": config, "sign": "style minus neutral",
                "layer_indexing": "zero-based decoder block outputs, before final model layer norm",
                "filter_results": str(args.filter_results) if args.filter_results else None,
                "require_neutral": args.require_neutral}, temporary)
    temporary.replace(target)
    print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    main()
