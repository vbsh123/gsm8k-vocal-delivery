
import argparse
import json
import random
from math import comb
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import librosa
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperModel
from peft import LoraConfig, get_peft_model
from scoring import score_response
import src.fast_dllm.diffa_generate as decoder

from src.configuration_DIFFA import DIFFAConfig
from src.modeling_DIFFA import DIFFAModel
from src.fast_dllm.model.modeling_llada import LLaDAModelLM

llm_id = "GSAI-ML/LLaDA-8B-Instruct"
whisper_id = "openai/whisper-large-v3"
diffa_id = "zhoujiaming777/DIFFA-2"
prompt = ("Solve the math problem in the audio. Give concise step-by-step reasoning "
              "with only the necessary calculations. If the question asks for a total, "
              "compute that total. End with one plain-text line in exactly this format: "
              "Final answer: <number>. Put a single numeric answer on that line, "
              "without Markdown, units, or separate component counts.")

def load_spoken_gsm(path="outputs/spoken-gsm-original"):
    """Yield each original recording's metadata, waveform, and sample rate."""
    path = Path(path)
    with (path / "manifest.jsonl").open() as manifest:
        for line in manifest:
            record = json.loads(line)
            audio, sample_rate = sf.read(path / record["audio_path"], dtype="float32")
            yield record, audio, sample_rate

def load_model():
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

    return model, feature_extractor, tokenizer

prefix_emotions = ["angry", "sad", "happy", "stressed", "afraid", "confident"]


def summarize_results(path):
    rows = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            rows[row["clip_id"], row["condition"]] = row
    print("Condition       Pairs  Accuracy  Baseline  Wins Losses  Exact p", flush=True)
    for emotion in prefix_emotions:
        pairs = [(r, rows[clip, "baseline"]) for (clip, condition), r in rows.items()
                 if condition == emotion and (clip, "baseline") in rows]
        if not pairs:
            continue
        n = len(pairs)
        wins = sum(r["correct"] and not b["correct"] for r, b in pairs)
        losses = sum(b["correct"] and not r["correct"] for r, b in pairs)
        d = wins + losses
        p = min(1.0, 2 * sum(comb(d, k) for k in range(min(wins, losses) + 1)) / 2**d)
        print(f"{emotion:15} {n:5} {sum(r['correct'] for r, b in pairs)/n:9.1%} "
              f"{sum(b['correct'] for r, b in pairs)/n:9.1%} {wins:5} {losses:6} {p:8.4f}", flush=True)


def accumulate(stats, emotion, baseline, framed):
    n = min(baseline.shape[1], framed.shape[1])
    base, changed = baseline[:, :n], framed[:, :n]
    difference = changed - base
    cosine = torch.nn.functional.cosine_similarity(base, changed, dim=-1)
    relative = difference.norm(dim=-1) / base.norm(dim=-1).clamp_min(1e-12)
    if emotion not in stats:
        stats[emotion] = {
            "sums": torch.zeros_like(difference),
            "counts": torch.zeros(n, dtype=torch.long),
            "cosine_sums": torch.zeros_like(cosine),
            "relative_difference_sums": torch.zeros_like(relative),
            "pair_mean_sum": torch.zeros_like(difference[:, 0]),
            "pair_count": 0,
        }
    entry = stats[emotion]
    extra = n - entry["counts"].shape[0]
    if extra > 0:
        for key in ("sums", "cosine_sums", "relative_difference_sums"):
            shape = list(entry[key].shape)
            shape[1] = extra
            entry[key] = torch.cat([entry[key], torch.zeros(shape)], dim=1)
        entry["counts"] = torch.cat([entry["counts"], torch.zeros(extra, dtype=torch.long)])
    entry["sums"][:, :n] += difference
    entry["counts"][:n] += 1
    entry["cosine_sums"][:, :n] += cosine
    entry["relative_difference_sums"][:, :n] += relative
    entry["pair_mean_sum"] += difference.mean(dim=1)
    entry["pair_count"] += 1


def filter_recordings(path, recordings):
    kept, skipped = [], []
    for recording in recordings:
        info = sf.info(path / recording["audio_path"])
        seconds = info.frames / info.samplerate
        if seconds > 30:
            skipped.append({
                "clip_id": recording["clip_id"],
                "question_id": recording["question_id"],
                "style": recording["style"],
                "audio_seconds": seconds,
                "reason": "recording_over_30_seconds",
            })
        else:
            kept.append(recording)
    return kept, skipped

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--path', type=Path, default=Path('outputs/spoken-gsm-original'))
    p.add_argument('--results-file', default='results-prompt-prefix.jsonl')
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--num-examples', type=int, default=300)
    args = p.parse_args(argv)
    if args.num_examples < 1:
        p.error('--num-examples must be positive')
    with (args.path / "manifest.jsonl").open() as manifest:
        recordings = [json.loads(line) for line in manifest if line.strip()]
    recordings, skipped = filter_recordings(args.path, recordings)
    recordings = recordings[:args.num_examples]
    with (args.path / (args.results_file + ".skipped.jsonl")).open("w") as f:
        for item in skipped:
            f.write(json.dumps(item) + "\n")
    print(f"Evaluating {len(recordings)} clips; skipped {len(skipped)} clips. "
            "Each kept clip fits within 30 seconds.", flush=True)
    if not recordings:
        print("No clips within 30 seconds; model was not loaded.")
        return
    results_path = args.path / args.results_file
    states_path = results_path.with_suffix('.hidden-states.pt')
    if states_path.exists():
        state = torch.load(states_path, map_location="cpu", weights_only=True)
    else:
        state = {"stats": {}, "completed_ids": [], "groups": ["acoustic", "semantic"],
                 "prompt": prompt, "emotions": prefix_emotions, "seed": 42,
                 "max_new_tokens": args.max_new_tokens, "steps": 128,
                 "sign": "prefixed minus baseline"}
    processed_ids = set(state["completed_ids"])
    pending = sum(r["clip_id"] not in processed_ids for r in recordings)
    print(f"{pending} clips remain; completed results will be skipped.", flush=True)
    if not pending:
        summarize_results(results_path)
        return
    model, fe, tokenizer = load_model()

    layers = model.llm_model.model.transformer.blocks
    bounds = {}
    captured = []
    original_insert = decoder.insert_audio_embedding

    def capture_insert(text_embeds, speech_embeds, audio_info_lengths):
        start = int(audio_info_lengths[0].item())
        bounds["start"] = start
        bounds["acoustic_end"] = start + model.config.prompt_size
        bounds["end"] = start + speech_embeds.shape[1]
        return original_insert(text_embeds, speech_embeds, audio_info_lengths)

    def make_hook(layer_index):
        def hook(module, inputs, output):
            hidden = output[0]
            acoustic = hidden[0, bounds["start"]:bounds["acoustic_end"]]
            semantic = hidden[0, bounds["acoustic_end"]:bounds["end"]]
            captured[layer_index].append(torch.stack([
                acoustic.float().mean(dim=0), semantic.float().mean(dim=0)
            ]).cpu())
        return hook

    def generate(text, inputs):
        nonlocal captured
        captured = [[] for _ in layers]
        bounds.clear()
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        with torch.inference_mode():
            response = model.generate(
                tokenizer,
                input_audio_features=inputs.input_features,
                input_audio_mask=inputs.attention_mask,
                system_prompt="", input_texts=[text], steps=128, block_length=32,
                max_new_tokens=args.max_new_tokens, temperature=0.0, accelerate="fast_dllm",
            )
        summaries = torch.stack([torch.stack(passes) for passes in captured])
        captured = []
        return response, summaries

    def save_response(recording, condition, text, response, summaries):
        row = {"clip_id": recording["clip_id"], "question_id": recording["question_id"],
               "condition": condition, "prompt": text, "response": response,
               "seed": 42, "forwards": summaries.shape[1],
               **score_response(response, recording["answer"])}
        with results_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"{recording['clip_id']} {condition}: correct={row['correct']} "
              f"answer={row['predicted_answer']} forwards={row['forwards']}", flush=True)

    eligible_ids = {r["clip_id"] for r in recordings}
    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    decoder.insert_audio_embedding = capture_insert
    try:
        for recording, audio, sample_rate in load_spoken_gsm(args.path):
            if recording["clip_id"] not in eligible_ids or recording["clip_id"] in processed_ids:
                continue
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            inputs = fe(audio, sampling_rate=16000, truncation=False,
                        return_tensors="pt", return_attention_mask=True).to("cuda")
            base_response, baseline = generate(prompt, inputs)
            save_response(recording, "baseline", prompt, base_response, baseline)
            clip_stats = {}
            for emo in prefix_emotions:
                text = 'Please note that speaker in the sound is ' + emo + '. ' + prompt
                response, framed = generate(text, inputs)
                save_response(recording, emo, text, response, framed)
                # Commit a question's aggregates only after all its conditions finish.
                accumulate(clip_stats, emo, baseline, framed)
                del framed
            for emo, entry in clip_stats.items():
                if emo not in state["stats"]:
                    state["stats"][emo] = entry
                    continue
                target = state["stats"][emo]
                extra = entry["counts"].shape[0] - target["counts"].shape[0]
                if extra > 0:
                    for key in ("sums", "cosine_sums", "relative_difference_sums"):
                        shape = list(target[key].shape)
                        shape[1] = extra
                        target[key] = torch.cat([target[key], torch.zeros(shape)], dim=1)
                    target["counts"] = torch.cat([target["counts"], torch.zeros(extra, dtype=torch.long)])
                n = entry["counts"].shape[0]
                for key in ("sums", "cosine_sums", "relative_difference_sums"):
                    target[key][:, :n] += entry[key]
                target["counts"][:n] += entry["counts"]
                target["pair_mean_sum"] += entry["pair_mean_sum"]
                target["pair_count"] += entry["pair_count"]
            state["completed_ids"].append(recording["clip_id"])
            del baseline, clip_stats
    finally:
        for handle in handles:
            handle.remove()
        decoder.insert_audio_embedding = original_insert
        torch.save(state, states_path)
        print("Saved hidden-state aggregates:", states_path, flush=True)
    summarize_results(results_path)


if __name__ == "__main__":
    main()
