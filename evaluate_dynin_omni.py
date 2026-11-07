"""Experimental spoken-math smoke using Dynin's official S2T token layout.

The released S2T examples transcribe. Here the instruction asks for solving;
inspect the printed responses before extending this to the full benchmark.
"""

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import wave

from compare_cloud_accuracy import PROMPT, read, summarize_deliveries
from scoring import score_response

REVISION = "d6e3ca874f399a976c42bd3c8a58eb469eb7e3a3"
MODEL = "snu-aidas/Dynin-Omni"
AUDIO_MODEL = "snu-aidas/emova_speech_tokenizer_hf"
RESULTS = "results-dynin-omni.jsonl"


def select_tasks(paths, styles, num_questions):
    """Use the same question/voice across requested styles, with complete audio."""
    groups = {}
    for folder in paths:
        for task in read(folder / "plan.jsonl"):
            if task["provider"] != "azure" or task["style"] not in styles:
                continue
            audio = folder / task["audio_path"]
            if not audio.exists():
                continue
            with wave.open(str(audio), "rb") as handle:
                duration = handle.getnframes() / handle.getframerate()
            if not 0 < duration <= 30:
                continue
            key = (task["question_id"], task["voice"])
            group = groups.setdefault(key, {})
            if task["style"] in group:
                raise ValueError(f"Duplicate question/voice/style: {key}, {task['style']}")
            group[task["style"]] = (folder, task)
    keys = sorted(k for k, group in groups.items() if all(s in group for s in styles))
    if num_questions:
        keys = keys[:num_questions]
    return [groups[key][style] for key in keys for style in styles]


def chat_instruction(instruction):
    # Upstream validation/speech.py: audio block precedes this chat fragment.
    return ("<|start_header_id|>user<|end_header_id|>\n"
            f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="+", type=Path,
                        default=[Path("outputs/azure-jenny-next500")])
    parser.add_argument("--styles", nargs="+", default=["neutral", "angry"])
    parser.add_argument("--num-questions", type=int, default=3, help="0 selects all complete groups.")
    parser.add_argument("--source", type=Path, default=Path("external/Dynin-Omni"))
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.num_questions < 0 or len(set(args.styles)) != len(args.styles):
        parser.error("Use nonnegative --num-questions and unique styles.")
    if (min(args.steps, args.max_new_tokens, args.block_length) <= 0
            or args.max_new_tokens % args.block_length
            or args.steps % (args.max_new_tokens // args.block_length)
            or args.steps > args.max_new_tokens):
        parser.error("Generation lengths/steps must satisfy upstream block divisibility.")
    selected = select_tasks(args.paths, args.styles, args.num_questions)
    if not selected:
        raise ValueError("No complete same-question/voice audio groups; check extracted plan.jsonl and WAVs.")
    folders = sorted({folder for folder, _ in selected})
    saved = {folder: {r["audio_path"]: r for r in read(folder / RESULTS)}
             if (folder / RESULTS).exists() else {} for folder in folders}
    pending = [(folder, task) for folder, task in selected
               if task["audio_path"] not in saved[folder]]
    print(f"Selected {len(selected)} clips ({len(selected)//len(args.styles)} matched questions); "
          f"{len(pending)} pending.", flush=True)
    recipe = dict(model=MODEL, audio_model=AUDIO_MODEL, source_revision=REVISION,
                  prompt=PROMPT, steps=args.steps, max_new_tokens=args.max_new_tokens,
                  block_length=args.block_length, temperature=0.0, seed=args.seed,
                  task="experimental_s2t_math_instruction")
    for folder in folders:
        settings = folder / "dynin-omni-settings.json"
        if settings.exists() and json.loads(settings.read_text()) != recipe:
            raise ValueError(f"Settings changed for {folder}; use a separate folder/result run.")
        if saved[folder] and not settings.exists():
            raise ValueError(f"Saved answers have no settings: {settings}")
    if args.prepare_only:
        for folder, task in selected:
            print(task["question_id"], task["style"], folder / task["audio_path"])
        return
    if not args.summary_only and pending:
        source = args.source.resolve()
        actual_revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if actual_revision != REVISION:
            raise ValueError("Upstream revision differs; run setup_dynin_omni.sh first.")
        sys.path.insert(0, str(source))
        # Match the upstream cache initialization before importing HF libraries.
        cache = Path(os.environ.get("DYNIN_OMNI_HF_CACHE_DIR") or os.environ.get("HF_HOME")
                     or source / "datasets/huggingface").expanduser().resolve()
        os.environ["HF_HOME"] = str(cache)
        os.environ["HF_HUB_CACHE"] = str(cache / "hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
        os.environ.pop("TRANSFORMERS_CACHE", None)
        (cache / "hub").mkdir(parents=True, exist_ok=True)
        import numpy as np
        import torch
        from transformers import AutoTokenizer
        from models import DyninOmniModelLM
        from models.modeling_emova_speech_tokenizer import EMOVASpeechTokenizer
        from training.prompting_utils import UniversalPrompting

        if not torch.cuda.is_available():
            raise RuntimeError("This smoke expects a CUDA GPU.")
        for folder in folders:
            (folder / "dynin-omni-settings.json").write_text(json.dumps(recipe, indent=2) + "\n")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left", trust_remote_code=True)
        prompting = UniversalPrompting(tokenizer, use_reserved_token=True)
        audio_model = EMOVASpeechTokenizer.from_pretrained(AUDIO_MODEL).cuda().eval()
        model = DyninOmniModelLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
        print(f"GPU memory after load: {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
        instruction_ids = tokenizer(chat_instruction(PROMPT), return_tensors="pt").input_ids.cuda()
        # Same offset as upstream validation/speech.py; these are discrete audio tokens.
        offset = len(prompting.text_tokenizer) + int(model.config.codebook_size)
        for index, (folder, task) in enumerate(pending, 1):
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            start = time.monotonic()
            with torch.inference_mode():
                audio_ids = audio_model.encode(str((folder / task["audio_path"]).resolve())).cuda() + offset
                specials = prompting.sptids_dict
                prefix = torch.cat([specials[name].cuda().unsqueeze(0)
                                    for name in ("<|s2t|>", "<|soa|>")], dim=1)
                input_ids = torch.cat([prefix, audio_ids,
                                       specials["<|eoa|>"].cuda().unsqueeze(0), instruction_ids], dim=1)
                output = model.mmu_generate(input_ids, max_new_tokens=args.max_new_tokens,
                                            steps=args.steps, block_length=args.block_length,
                                            temperature=0.0, remasking="low_confidence")
                generated = output[0, input_ids.shape[1]:].tolist()
                # Upstream returns a full fixed-length suffix, so stop before the first EOS/EOT.
                stops = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")}
                stop = next((i for i, token in enumerate(generated) if token in stops), len(generated))
                response = tokenizer.decode(generated[:stop], skip_special_tokens=True).strip()
            score = score_response(response, task["answer"])
            row = {**{key: task[key] for key in ("audio_path", "question_id", "style", "voice")},
                   **recipe, **score, "response": response,
                   "audio_tokens": audio_ids.shape[1], "seconds": time.monotonic()-start}
            with (folder / RESULTS).open("a") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            saved[folder][task["audio_path"]] = row
            print(f"\n[{index}/{len(pending)}] {task['question_id']} {task['style']} "
                  f"correct={score['correct']} seconds={row['seconds']:.1f}\n{response}\n", flush=True)
    tasks = [{**task, "audio_path": str(folder.resolve() / task["audio_path"])}
             for folder, task in selected]
    answers = {str(folder.resolve() / name): row for folder in folders for name, row in saved[folder].items()}
    summarize_deliveries(tasks, answers, args.styles)
    print("Experimental speech instruction-following smoke: inspect answers, not only scores.")


if __name__ == "__main__":
    main()
