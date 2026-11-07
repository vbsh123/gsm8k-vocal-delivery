"""Evaluate Jenny spoken math with Qwen2.5-Omni-7B, audio input and text output.

Defaults to a three-question smoke. --num-questions 0 evaluates all eligible
clips independently; paired reports then intersect questions for each delivery
comparison. Emotion labels do not restrict generation.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time
import wave

from cloud_tts_smoke import STYLES, AZURE_EXTRA_STYLES
from compare_cloud_accuracy import PROMPT, read, summarize_deliveries
from scoring import score_response

MODEL = "Qwen/Qwen2.5-Omni-7B"
REVISION = "ae9e1690543ffd5c0221dc27f79834d0294cba00"
RESULTS = "results-qwen-omni.jsonl"
SETTINGS = "qwen-omni-settings.json"
SYSTEM_PROMPT = "You are a helpful assistant."


def select_tasks(paths, styles, num_questions):
    groups = {}
    for folder in paths:
        for task in read(folder / "plan.jsonl"):
            if task["provider"] != "azure" or task["style"] not in styles:
                continue
            audio = folder / task["audio_path"]
            if not audio.exists():
                raise FileNotFoundError(f"Extract the audio archive first: {audio}")
            with wave.open(str(audio), "rb") as handle:
                duration = handle.getnframes() / handle.getframerate()
            if not 0 < duration <= 30:
                continue
            key = (task["question_id"], task["voice"])
            group = groups.setdefault(key, {})
            if task["style"] in group:
                raise ValueError(f"Duplicate question/voice/style: {key}, {task['style']}")
            group[task["style"]] = (folder, task)
    keys = sorted(groups)
    if num_questions:
        # A small smoke compares exactly the same questions in every style.
        keys = [k for k in keys if all(s in groups[k] for s in styles)][:num_questions]
    return [groups[key][style] for key in keys for style in styles if style in groups[key]]


def conversation():
    # Neither the question transcript, reference answer, nor intended emotion is
    # included. The processor gets the actual waveform separately.
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": PROMPT}]}]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", nargs="+", type=Path, default=[Path("outputs/azure-jenny-next500")])
    p.add_argument("--styles", nargs="+", choices=STYLES+AZURE_EXTRA_STYLES,
                   default=["neutral", "angry", "happy", "sad", "terrified", "shouting"])
    p.add_argument("--num-questions", type=int, default=3,
                   help="Smoke: up to N complete same-question/voice groups. 0: all eligible clips.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    if args.num_questions < 0 or args.max_new_tokens < 1 or len(set(args.styles)) != len(args.styles):
        p.error("Use nonnegative question count, positive token limit, and unique styles.")
    args.paths = [folder.resolve() for folder in args.paths]
    selected = select_tasks(args.paths, args.styles, args.num_questions)
    if not selected:
        p.error("No eligible recordings. Check paths/styles; the small smoke requires complete style groups.")
    folders = sorted({folder for folder, _ in selected})
    saved = {folder: {r["audio_path"]: r for r in read(folder/RESULTS)}
             if (folder/RESULTS).exists() else {} for folder in folders}
    recipe = dict(model=MODEL, revision=REVISION, transformers="4.52.4", prompt=PROMPT,
                  system_prompt=SYSTEM_PROMPT, max_new_tokens=args.max_new_tokens,
                  seed=args.seed, do_sample=False, dtype="bfloat16", attention="sdpa",
                  sample_rate=16000, max_audio_seconds=30, return_audio=False)
    for folder in folders:
        settings = folder/SETTINGS
        if settings.exists() and json.loads(settings.read_text()) != recipe:
            p.error(f"Settings differ in {settings}; use a separate result folder for a new recipe.")
        if saved[folder] and not settings.exists():
            p.error(f"Answers exist without settings: {settings}")
    pending = [(folder, task) for folder, task in selected if task["audio_path"] not in saved[folder]]
    n_questions = len({(task["question_id"], task["voice"]) for _, task in selected})
    print(f"Selected {len(selected)} clips from {n_questions} question/voice groups; "
          f"{len(pending)} pending. All requested styles, without emotion filtering.", flush=True)
    print("Clips by delivery:", dict(Counter(task["style"] for _, task in selected)), flush=True)
    if args.prepare_only:
        for folder, task in selected:
            print(task["question_id"], task["style"], folder/task["audio_path"])
        return
    if pending and not args.summary_only:
        # Avoid inherited cache paths such as /models; keep downloaded assets
        # under the workspace unless the caller deliberately chose HF_HOME.
        cache = Path(os.environ.get("HF_HOME") or "outputs/hf-qwen-omni").expanduser().resolve()
        os.environ["HF_HOME"] = str(cache)
        os.environ["HF_HUB_CACHE"] = str(cache/"hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache/"hub")
        os.environ.pop("TRANSFORMERS_CACHE", None)
        (cache/"hub").mkdir(parents=True, exist_ok=True)
        import librosa
        import numpy as np
        import soundfile as sf
        import torch
        import transformers
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        if transformers.__version__ != recipe["transformers"]:
            raise RuntimeError("Run setup_qwen_omni.sh and use the qwen-omni environment.")
        if not torch.cuda.is_available():
            raise RuntimeError("This evaluation expects a CUDA GPU.")
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL, revision=REVISION)
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
            device_map="cuda:0", attn_implementation="sdpa").eval()
        model.disable_talker()
        torch.cuda.empty_cache()
        print(f"Loaded Qwen; GPU allocated: {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
        for folder in folders:
            (folder/SETTINGS).write_text(json.dumps(recipe, indent=2)+"\n")
        text = processor.apply_chat_template(conversation(), tokenize=False, add_generation_prompt=True)
        for index, (folder, task) in enumerate(pending, 1):
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            start = time.monotonic()
            audio, rate = sf.read(folder/task["audio_path"], dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if rate != 16000:
                audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
            inputs = processor(text=text, audio=[audio], sampling_rate=16000,
                               return_tensors="pt", padding=True).to(model.device).to(model.dtype)
            if "input_features" not in inputs or "feature_attention_mask" not in inputs:
                raise RuntimeError("Processor did not produce audio features and their mask.")
            audio_token_index = model.config.thinker_config.audio_token_index
            if not (inputs.input_ids == audio_token_index).any().item():
                raise RuntimeError("Chat template did not insert audio tokens.")
            print(f"[{index}/{len(pending)}] {task['question_id']} {task['style']}", flush=True)
            with torch.inference_mode():
                generated = model.generate(**inputs, return_audio=False,
                                           thinker_max_new_tokens=args.max_new_tokens,
                                           thinker_do_sample=False)
            suffix = generated[:, inputs.input_ids.shape[1]:]
            response = processor.batch_decode(suffix, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0].strip()
            score = score_response(response, str(task["answer"]).rsplit("####", 1)[-1].strip())
            row = {**{key: task[key] for key in ("audio_path", "question_id", "voice", "style")},
                   **recipe, **score, "response": response, "output_tokens": suffix.shape[1],
                   "generation_seconds": time.monotonic()-start}
            with (folder/RESULTS).open("a") as handle:
                handle.write(json.dumps(row)+"\n")
                handle.flush()
            saved[folder][task["audio_path"]] = row
            print(f"  correct={score['correct']} answer={score['predicted_answer']} "
                  f"seconds={row['generation_seconds']:.1f}\n{response}\n", flush=True)
    tasks = [{**task, "audio_path": str(folder/task["audio_path"])} for folder, task in selected]
    answers = {str(folder/name): row for folder in folders for name, row in saved[folder].items()}
    summarize_deliveries(tasks, answers, args.styles)
    print(f"Saved answers: {RESULTS} in each source folder. Existing DIFFA/Dynin answers are separate.")


if __name__ == "__main__":
    main()
