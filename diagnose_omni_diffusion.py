"""Small Omni controls: same question as text math, audio math, and transcription.

Compare the existing text-only output mask with native decoding. Results are
diagnostics, stored separately from the delivery benchmark, and never launch it.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import time
from unittest.mock import patch

from compare_cloud_accuracy import PROMPT, read
from evaluate_omni_diffusion import MODEL, REVISION, TEMPLATE, assets, insert_audio_ids
from evaluate_qwen_omni import select_tasks
from scoring import score_response


def word_error_rate(reference, hypothesis):
    ref, hyp = [re.findall(r"\w+", s.lower()) for s in (reference, hypothesis)]
    if not ref:
        return None
    previous = list(range(len(hyp)+1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(a != b)))
        previous = current
    return previous[-1]/len(ref)


def instruction(mode, question):
    if mode == "text_math":
        return PROMPT.replace("in the audio", "below")+"\n\n"+question
    if mode == "audio_math":
        return PROMPT
    if mode == "transcribe":
        return "Convert the speech to text."
    raise ValueError(mode)


def report(rows):
    print("\nDiagnostic results (single-question smoke by default)")
    for row in rows:
        value = f"WER={row['word_error_rate']:.1%}" if row["mode"] == "transcribe" else f"correct={row['correct']} answer={row['predicted_answer']}"
        print(f"{row['question_id']} {row['mode']:12} {row['decoding']:9} "
              f"{value}; repeated 4-grams={row['repetition_fraction']:.1%}; "
              f"EOS={row['eos_seen']}; nontext tokens={row['nontext_tokens']}")
    print("WER uses simple lowercase word normalization; listen/check fractions before treating mismatches as audio errors.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", type=Path, default=Path("outputs/azure-jenny500-neutral"))
    p.add_argument("--num-questions", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("outputs/omni-diagnostics"))
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    result_path = args.output/"results.jsonl"
    if args.summary_only:
        report(read(result_path) if result_path.exists() else [])
        return
    if args.num_questions < 1 or not 0 < args.steps <= args.max_new_tokens:
        p.error("Use positive question count and 0 < steps <= max-new-tokens")
    tasks = [task for _, task in select_tasks([args.path.resolve()], ["neutral"], args.num_questions)]
    if not tasks:
        p.error("No eligible neutral audio")
    jobs = [(task, mode, decoding) for task in tasks
            for mode in ("text_math", "audio_math", "transcribe")
            for decoding in ("masked", "native")]
    recipe = dict(model=MODEL, revision=REVISION, steps=args.steps, max_new_tokens=args.max_new_tokens,
                  seed=args.seed, algorithm="entropy", temperature=0.0, top_p=1.0,
                  tasks=tasks, source=str(args.path.resolve()))
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output/"config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != recipe:
        p.error("Diagnostic settings changed; use a new --output")
    rows = read(result_path) if result_path.exists() else []
    completed = {(r["question_id"], r["mode"], r["decoding"]) for r in rows}
    pending = [job for job in jobs if (job[0]["question_id"], job[1], job[2]) not in completed]
    print(f"{len(tasks)} questions; {len(jobs)} diagnostic generations; {len(pending)} pending.", flush=True)
    if args.prepare_only:
        return
    if not pending:
        report(rows)
        return
    path, config, tokenizer, model_class = assets()
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from funasr.utils.load_utils import extract_fbank
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    with patch("funasr.utils.install_model_requirements.install_requirements", return_value=True):
        model, info = model_class.from_pretrained(path, config=config, torch_dtype=torch.bfloat16,
            device_map="cuda:0", attn_implementation="sdpa", output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise RuntimeError(f"Incomplete checkpoint load: {info}")
    model.eval()
    model.generation_config.mask_token_id = config.mask_token_id
    frontend = model.model.audio_model.kwargs["frontend"]
    token = lambda name: tokenizer.encode(name, add_special_tokens=False)[0]
    cutoff = min(token("<|audio_0|>"), token("<|image_0|>"))
    def text_only(step, sequence, logits):
        logits[..., cutoff:] = -float("inf")
        return logits
    config_path.write_text(json.dumps(recipe, indent=2)+"\n")
    features_cache = {}
    for index, (task, mode, decoding) in enumerate(pending, 1):
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        question = task["question"]
        content = instruction(mode, question)
        if mode == "text_math":
            prompt = "<|im_start|>user\n"+content+"<|im_end|>\n<|im_start|>assistant\n"
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            audio_args = {}
        else:
            if task["audio_path"] not in features_cache:
                audio, rate = sf.read(args.path/task["audio_path"], dtype="float32")
                if audio.ndim == 2: audio = audio.mean(axis=1)
                if rate != 16000: audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
                feature_batch, _ = extract_fbank(torch.from_numpy(audio), data_type="sound", frontend=frontend)
                features_cache[task["audio_path"]] = feature_batch[0]
            features = features_cache[task["audio_path"]]
            prompt = TEMPLATE.format(content)
            ids, positions = insert_audio_ids(tokenizer.encode(prompt, add_special_tokens=False),
                token("<|audio|>"), token("<|begin_of_audio|>"), token("<|context_of_audio|>"),
                token("<|end_of_audio|>"), features.shape[0]+4)
            indices = torch.stack([torch.zeros((1,len(positions)), dtype=torch.long),
                                   torch.tensor([positions], dtype=torch.long)], dim=0)
            audio_args = dict(audios=[features], audio_indices=[indices])
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
        kwargs = {"generation_logits_hook_func": text_only} if decoding == "masked" else {}
        started = time.monotonic()
        print(f"[{index}/{len(pending)}] {task['question_id']} {mode} {decoding}", flush=True)
        with torch.inference_mode():
            output, histories = model.generate(input_ids, **audio_args, **kwargs,
                max_new_tokens=args.max_new_tokens, steps=args.steps, temperature=0.0,
                top_p=1.0, alg="entropy", alg_temp=0.0, cfg=0.0, block_size=args.max_new_tokens)
            suffix = output[0,len(ids):].tolist()
            stops = {tokenizer.eos_token_id, token("<|im_end|>")}
            stop = next((i for i,x in enumerate(suffix) if x in stops),len(suffix))
            response = tokenizer.decode(suffix[:stop],skip_special_tokens=True).strip()
            raw = tokenizer.decode(suffix[:stop],skip_special_tokens=False)
        del output, histories
        words = response.split(); grams = [tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
        repeated = sum(n-1 for n in Counter(grams).values())
        row = dict(question_id=task["question_id"], audio_path=task["audio_path"], mode=mode,
            decoding=decoding, seed=args.seed, prompt=prompt, question=question, response=response,
            raw_response=raw, generation_seconds=time.monotonic()-started,
            eos_seen=stop<len(suffix), nontext_tokens=sum(x>=cutoff for x in suffix[:stop]),
            repetition_fraction=repeated/len(grams) if grams else 0,
            word_error_rate=word_error_rate(question,response) if mode=="transcribe" else None,
            **score_response(response,str(task["answer"]).rsplit("####",1)[-1].strip()))
        with result_path.open("a") as stream:
            stream.write(json.dumps(row)+"\n"); stream.flush()
        rows.append(row)
        print(response+"\n",flush=True)
    report(rows)


if __name__ == "__main__":
    main()
