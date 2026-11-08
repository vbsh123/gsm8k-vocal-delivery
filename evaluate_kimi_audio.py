"""Kimi-Audio spoken math: audio + fixed text instruction -> text-only answer.

Default: three matched questions in neutral/angry/happy/sad. Use
--num-questions 0 for all eligible clips, without emotion/correctness filtering.
Saved answers are separate from Qwen/DIFFA/Omni and resumable per recording.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from unittest.mock import patch

from cloud_tts_smoke import STYLES, AZURE_EXTRA_STYLES
from compare_cloud_accuracy import PROMPT, read, summarize_deliveries
from evaluate_qwen_omni import select_tasks
from scoring import score_response

MODEL = "moonshotai/Kimi-Audio-7B-Instruct"
REVISION = "9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"
SOURCE_REVISION = "349251e1d8f4f98d58fda59246381faecd7392e0"
TOKENIZER_MODEL = "THUDM/glm-4-voice-tokenizer"
TOKENIZER_REVISION = "a5f2404e63c84e92f5238908e1706316324ebafa"
RESULTS = "results-kimi-audio.jsonl"
SETTINGS = "kimi-audio-settings.json"


def messages(audio_path):
    # Same user role for instruction + audio follows the official Kimi API.
    # Do not include the question transcript, reference answer, or style label.
    return [{"role": "user", "message_type": "text", "content": PROMPT},
            {"role": "user", "message_type": "audio", "content": str(audio_path)}]


def assets(source):
    source = source.resolve()
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != SOURCE_REVISION:
        raise ValueError("Kimi source differs from the pinned version; run setup_kimi_audio.sh")
    sys.path.insert(0, str(source))
    cache = Path(os.environ.get("HF_HOME") or "outputs/hf-kimi-audio").expanduser().resolve()
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache/"hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache/"hub")
    os.environ.pop("TRANSFORMERS_CACHE", None)
    (cache/"hub").mkdir(parents=True, exist_ok=True)
    import transformers
    if transformers.__version__ != "4.51.3":
        raise RuntimeError("Use the kimi-audio environment from setup_kimi_audio.sh")
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from kimia_infer.api.kimia import KimiAudio
    print("Checking/downloading Kimi and input encoders; speech-output weights are excluded.", flush=True)
    model_path = snapshot_download(MODEL, revision=REVISION,
                                  ignore_patterns=["audio_detokenizer/*", "vocoder/*"])
    tokenizer_path = snapshot_download(TOKENIZER_MODEL, revision=TOKENIZER_REVISION)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], model_path)
    if not (Path(model_path)/"whisper-large-v3/config.json").exists():
        raise ValueError("Kimi snapshot is missing its continuous Whisper input encoder")
    print("Kimi source/import/tokenizer preflight passed; no model loaded onto GPU yet.", flush=True)
    return model_path, tokenizer_path, KimiAudio


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", type=Path, nargs="+", default=[
        Path("outputs/azure-jenny500-extra"), Path("outputs/azure-jenny500-neutral")])
    p.add_argument("--styles", nargs="+", choices=STYLES+AZURE_EXTRA_STYLES,
                   default=["neutral", "angry", "happy", "sad"])
    p.add_argument("--num-questions", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source", type=Path, default=Path("external/Kimi-Audio"))
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    if args.download_only:
        assets(args.source)
        return
    if args.num_questions < 0 or not 0 < args.max_new_tokens <= 4096:
        p.error("Use nonnegative question count and 1..4096 output tokens (upstream buffer limit)")
    if len(set(args.styles)) != len(args.styles):
        p.error("Styles must be unique")
    selected = select_tasks([folder.resolve() for folder in args.paths], args.styles, args.num_questions)
    if not selected:
        p.error("No eligible clips; small smoke needs complete same-question/voice style groups")
    folders = sorted({folder for folder, _ in selected})
    saved = {folder: {r["audio_path"]: r for r in read(folder/RESULTS)}
             if (folder/RESULTS).exists() else {} for folder in folders}
    recipe = dict(model=MODEL, revision=REVISION, source_revision=SOURCE_REVISION,
                  tokenizer_model=TOKENIZER_MODEL, tokenizer_revision=TOKENIZER_REVISION,
                  transformers="4.51.3", prompt=PROMPT, max_new_tokens=args.max_new_tokens,
                  seed=args.seed, text_temperature=0.0, audio_temperature=0.0,
                  text_top_k=5, audio_top_k=5, text_repetition_penalty=1.0,
                  audio_repetition_penalty=1.0, output_type="text", load_detokenizer=False)
    for folder in folders:
        if (folder/SETTINGS).exists() and json.loads((folder/SETTINGS).read_text()) != recipe:
            p.error(f"Settings changed in {folder/SETTINGS}; use a separate result folder")
        if saved[folder] and not (folder/SETTINGS).exists():
            p.error(f"Answers exist without settings: {folder/SETTINGS}")
    pending = [(folder, task) for folder, task in selected if task["audio_path"] not in saved[folder]]
    print(f"Selected {len(selected)} clips; {len(pending)} pending. Emotion labels do not filter generation.", flush=True)
    print("By requested delivery:", dict(Counter(task["style"] for _, task in selected)), flush=True)
    if args.prepare_only:
        return
    if pending and not args.summary_only:
        model_path, tokenizer_path, KimiAudio = assets(args.source)
        import numpy as np
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        from kimia_infer.api.prompt_manager import Glm4Tokenizer
        # Preserve the official tokenizer implementation; give its constructor
        # the pinned local snapshot instead of the upstream hardcoded main ID.
        with patch("kimia_infer.api.prompt_manager.Glm4Tokenizer",
                   side_effect=lambda _: Glm4Tokenizer(tokenizer_path)):
            model = KimiAudio(model_path=model_path, load_detokenizer=False)
        model.alm.eval()
        for folder in folders:
            (folder/SETTINGS).write_text(json.dumps(recipe, indent=2)+"\n")
        print(f"Kimi loaded; allocated GPU memory {torch.cuda.memory_allocated()/2**30:.2f} GiB.", flush=True)
        for index, (folder, task) in enumerate(pending, 1):
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            print(f"[{index}/{len(pending)}] {task['question_id']} {task['style']}", flush=True)
            started = time.monotonic()
            with torch.inference_mode():
                _, response = model.generate(messages((folder/task["audio_path"]).resolve()),
                    output_type="text", max_new_tokens=args.max_new_tokens,
                    text_temperature=0.0, audio_temperature=0.0, text_top_k=5, audio_top_k=5,
                    text_repetition_penalty=1.0, audio_repetition_penalty=1.0)
            score = score_response(response, str(task["answer"]).rsplit("####",1)[-1].strip())
            row = {**{k: task[k] for k in ("audio_path", "question_id", "voice", "style")},
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
    print("Inspect the smoke responses before launching the full benchmark.")


if __name__ == "__main__":
    main()
