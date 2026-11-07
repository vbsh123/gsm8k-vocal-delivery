"""Omni-Diffusion spoken math with continuous SenseVoice audio features and text output.

Uses the released checkpoint's model/generation code. Audio preprocessing follows
the contiguous input path in VITA-MLLM/Omni-Diffusion; speech/image generation
components are unnecessary. --smoke-gate checks answer format, not correctness.
"""

import argparse
import json
import os
from pathlib import Path
import random
import time

from compare_cloud_accuracy import PROMPT, read, summarize_deliveries
from evaluate_qwen_omni import select_tasks
from scoring import score_response

MODEL = "lijiang/Omni-Diffusion"
REVISION = "2233ee1ae2aa2b63e00466197afdeea0ca9e0901"
RESULTS = "results-omni-diffusion.jsonl"
SETTINGS = "omni-diffusion-settings.json"
TEMPLATE = "<|im_start|>user\n{}\n<|audio|><|im_end|>\n<|im_start|>assistant\n"


def cache_environment():
    cache = Path(os.environ.get("HF_HOME") or "outputs/hf-omni-diffusion").expanduser().resolve()
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache/"hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache/"hub")
    os.environ.pop("TRANSFORMERS_CACHE", None)
    (cache/"hub").mkdir(parents=True, exist_ok=True)


def assets():
    cache_environment()
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    print("Downloading/checking public Omni-Diffusion weights and SenseVoice assets (CPU only).", flush=True)
    path = snapshot_download(MODEL, revision=REVISION)
    snapshot_download("FunAudioLLM/SenseVoiceSmall")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model_class = get_class_from_dynamic_module(config.auto_map["AutoModel"], path)
    for name in ("<|audio|>", "<|begin_of_audio|>", "<|context_of_audio|>",
                 "<|end_of_audio|>", "<|audio_0|>", "<|image_0|>"):
        ids = tokenizer.encode(name, add_special_tokens=False)
        if len(ids) != 1 or ids[0] == tokenizer.unk_token_id:
            raise ValueError(f"Missing single multimodal token: {name}")
    if not isinstance(config.mask_token_id, int):
        raise ValueError("Checkpoint has no valid mask token")
    print("Checkpoint/import/tokenizer preflight passed.", flush=True)
    return path, config, tokenizer, model_class


def insert_audio_ids(ids, tag, start, context, end, length):
    if ids.count(tag) != 1:
        raise ValueError("Expected one audio placeholder")
    position = ids.index(tag)
    # SenseVoice adds four language/emotion/event/normalization query frames.
    indices = list(range(position+1, position+1+length))
    return ids[:position]+[start]+[context]*length+[end]+ids[position+1:], indices


def smoke_gate(selected, saved):
    rows = [saved[folder][task["audio_path"]] for folder, task in selected
            if task["audio_path"] in saved[folder]]
    parsed = sum(not score_response(r["response"], r["reference_answer"])["parse_failed"] for r in rows)
    print(f"Smoke format gate: {parsed}/{len(selected)} responses contain a parseable Final answer.", flush=True)
    if len(rows) != len(selected) or parsed*2 < len(selected):
        raise SystemExit("Smoke did not produce enough formatted answers (requires at least half). "
                         "Full evaluation was NOT started; inspect saved responses/log.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", type=Path, nargs="+", default=[
        Path("outputs/azure-jenny500-extra"), Path("outputs/azure-jenny500-neutral")])
    p.add_argument("--styles", nargs="+", default=["neutral", "angry", "happy", "sad"])
    p.add_argument("--num-questions", type=int, default=3, help="0: all eligible clips; positive: complete style groups.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--smoke-gate", action="store_true")
    args = p.parse_args()
    if args.download_only:
        assets()
        return
    if args.num_questions < 0 or min(args.max_new_tokens, args.steps) < 1 or args.steps > args.max_new_tokens:
        p.error("Use a nonnegative question count and 0 < steps <= max-new-tokens")
    if len(set(args.styles)) != len(args.styles):
        p.error("Styles must be unique")
    paths = [folder.resolve() for folder in args.paths]
    selected = select_tasks(paths, args.styles, args.num_questions)
    if not selected:
        p.error("No eligible clips; check extracted audio and plan.jsonl files")
    folders = sorted({folder for folder, _ in selected})
    saved = {folder: {r["audio_path"]: r for r in read(folder/RESULTS)}
             if (folder/RESULTS).exists() else {} for folder in folders}
    recipe = dict(model=MODEL, revision=REVISION, transformers="4.51.3", prompt=PROMPT,
                  max_new_tokens=args.max_new_tokens, steps=args.steps, seed=args.seed,
                  temperature=0.0, top_p=1.0, algorithm="entropy", attention="sdpa",
                  dtype="bfloat16", input="continuous SenseVoice fbank; no transcript",
                  text_only_logit_mask=True, chat_template=TEMPLATE)
    for folder in folders:
        if (folder/SETTINGS).exists() and json.loads((folder/SETTINGS).read_text()) != recipe:
            p.error(f"Settings differ in {folder/SETTINGS}; use a separate result folder")
        if saved[folder] and not (folder/SETTINGS).exists():
            p.error(f"Answers exist without {folder/SETTINGS}")
    pending = [(folder, task) for folder, task in selected if task["audio_path"] not in saved[folder]]
    print(f"Selected {len(selected)} clips; {len(pending)} pending.", flush=True)
    if args.prepare_only:
        return
    if pending and not args.summary_only:
        path, config, tokenizer, model_class = assets()
        import librosa
        import numpy as np
        import soundfile as sf
        import torch
        import transformers
        from funasr.utils.load_utils import extract_fbank
        if transformers.__version__ != recipe["transformers"]:
            raise RuntimeError("Use the omni-diffusion environment created by setup_omni_diffusion.sh")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        # The checkpoint overrides from_pretrained. Request its supported tuple
        # return explicitly to avoid its fallback loading the entire model twice.
        model, info = model_class.from_pretrained(
            path, config=config, torch_dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa", output_loading_info=True)
        model.eval()
        if info.get("missing_keys") or info.get("mismatched_keys"):
            raise RuntimeError(f"Incomplete checkpoint load: {info}")
        model.generation_config.mask_token_id = config.mask_token_id
        encoder = model.model.audio_model
        frontend = encoder.kwargs["frontend"]
        token = lambda name: tokenizer.encode(name, add_special_tokens=False)[0]
        raw_ids = tokenizer.encode(TEMPLATE.format(PROMPT), add_special_tokens=False)
        cutoff = min(token("<|audio_0|>"), token("<|image_0|>"))

        def text_only(step, sequence, logits):
            # Restrict output modality, preserving the pretrained continuous audio input path.
            logits[..., cutoff:] = -float("inf")
            return logits

        for folder in folders:
            (folder/SETTINGS).write_text(json.dumps(recipe, indent=2)+"\n")
        print(f"Model loaded; allocated GPU memory {torch.cuda.memory_allocated()/2**30:.2f} GiB.", flush=True)
        for index, (folder, task) in enumerate(pending, 1):
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            started = time.monotonic()
            audio, rate = sf.read(folder/task["audio_path"], dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if rate != 16000:
                audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
            features, _ = extract_fbank(torch.from_numpy(audio), data_type="sound", frontend=frontend)
            features = features[0]
            ids, positions = insert_audio_ids(raw_ids, token("<|audio|>"), token("<|begin_of_audio|>"),
                token("<|context_of_audio|>"), token("<|end_of_audio|>"), features.shape[0]+4)
            input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
            indices = torch.stack([torch.zeros((1,len(positions)), dtype=torch.long),
                                   torch.tensor([positions], dtype=torch.long)], dim=0)
            print(f"[{index}/{len(pending)}] {task['question_id']} {task['style']}", flush=True)
            with torch.inference_mode():
                output, histories = model.generate(
                    input_ids, audios=[features], audio_indices=[indices],
                    max_new_tokens=args.max_new_tokens, steps=args.steps, temperature=0.0,
                    top_p=1.0, alg="entropy", alg_temp=0.0, cfg=0.0,
                    block_size=args.max_new_tokens, generation_logits_hook_func=text_only)
                suffix = output[0, len(ids):].tolist()
                stops = {tokenizer.eos_token_id, token("<|im_end|>")}
                stop = next((i for i, value in enumerate(suffix) if value in stops), len(suffix))
                response = tokenizer.decode(suffix[:stop], skip_special_tokens=True).strip()
            del output, histories
            score = score_response(response, str(task["answer"]).rsplit("####",1)[-1].strip())
            row = {**{k: task[k] for k in ("question_id", "voice", "style", "audio_path")},
                   **recipe, **score, "response": response, "generation_seconds": time.monotonic()-started}
            with (folder/RESULTS).open("a") as handle:
                handle.write(json.dumps(row)+"\n")
                handle.flush()
            saved[folder][task["audio_path"]] = row
            print(f"  correct={score['correct']} answer={score['predicted_answer']} "
                  f"seconds={row['generation_seconds']:.1f}\n{response}\n", flush=True)
    tasks = [{**task,"audio_path":str(folder/task["audio_path"])} for folder,task in selected]
    results = {str(folder/name):row for folder in folders for name,row in saved[folder].items()}
    summarize_deliveries(tasks, results, args.styles)
    if args.smoke_gate:
        smoke_gate(selected, saved)


if __name__ == "__main__":
    main()
