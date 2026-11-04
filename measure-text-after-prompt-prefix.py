"""Compare text-only LLaDA GSM8K accuracy with and without emotional framing."""

import argparse
import json
import random
from math import comb
from pathlib import Path
from types import SimpleNamespace

from scoring import score_response


MODEL = "GSAI-ML/LLaDA-8B-Instruct"
EMOTIONS = ["neutral", "angry", "sad", "happy", "stressed", "afraid", "confident"]
INSTRUCTION = (
    "Solve the following math problem. Give concise step-by-step reasoning "
    "with only the necessary calculations. If the question asks for a total, "
    "compute that total. End with one plain-text line in exactly this format: "
    "Final answer: <number>. Put a single numeric answer on that line, "
    "without Markdown, units, or separate component counts."
)


def make_prompt(question, condition):
    prefix = "" if condition == "baseline" else f"The person asking this question feels {condition}. "
    return prefix + INSTRUCTION + "\n\n" + question


def read_results(path):
    if not path.exists():
        return {}
    with path.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {(r["question_id"], r["condition"]): r for r in rows}


def summarize(rows):
    print("Condition       Pairs  Accuracy  Baseline  Wins Losses  Exact p", flush=True)
    for emotion in EMOTIONS:
        pairs = [(r, rows[qid, "baseline"]) for (qid, condition), r in rows.items()
                 if condition == emotion and (qid, "baseline") in rows]
        if not pairs:
            continue
        n = len(pairs)
        wins = sum(r["correct"] and not b["correct"] for r, b in pairs)
        losses = sum(b["correct"] and not r["correct"] for r, b in pairs)
        d = wins + losses
        p = min(1.0, 2 * sum(comb(d, k) for k in range(min(wins, losses) + 1)) / 2**d)
        print(f"{emotion:15} {n:5} {sum(r['correct'] for r, b in pairs)/n:9.1%} "
              f"{sum(b['correct'] for r, b in pairs)/n:9.1%} {wins:5} {losses:6} {p:8.4f}", flush=True)
    print("Parse failures count as incorrect. P-values are unadjusted.", flush=True)


def load_questions(output, num_examples):
    path = output / "questions.jsonl"
    if path.exists():
        with path.open() as stream:
            return [json.loads(line) for line in stream if line.strip()][:num_examples]

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    source = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
    rows = pq.read_table(source).to_pylist()
    questions = [{"question_id": f"gsm8k_test_{i:05d}", "question": r["question"],
                  "answer": r["answer"].rsplit("####", 1)[-1].strip()}
                 for i, r in enumerate(rows)]
    path.write_text("".join(json.dumps(r) + "\n" for r in questions))
    return questions[:num_examples]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-examples", type=int, default=300)
    p.add_argument("--output", type=Path, default=Path("outputs/text-prompt-prefix300"))
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args(argv)
    if args.num_examples < 1:
        p.error("--num-examples must be positive")
    if args.max_new_tokens < 32 or args.max_new_tokens % 32 or 128 % (args.max_new_tokens // 32):
        p.error("--max-new-tokens must be a multiple of 32 with its block count dividing 128")
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.jsonl"
    results = read_results(results_path)
    if args.summary_only:
        summarize(results)
        return
    questions = load_questions(args.output, args.num_examples)
    conditions = ["baseline", *EMOTIONS]
    pending = sum((q["question_id"], c) not in results for q in questions for c in conditions)
    print(f"{len(questions)} questions, {len(conditions)} conditions; {pending} generations pending.", flush=True)
    if not pending:
        summarize(results)
        return

    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from src.fast_dllm.model.modeling_llada import LLaDAModelLM
    from src.fast_dllm.diffa_generate import generate

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLaDAModelLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    llm.requires_grad_(False)
    # Reuse our fast-dLLM decoder with zero audio positions: only text enters LLaDA.
    model = SimpleNamespace(llm_model=llm, device=llm.device)
    empty_audio = torch.empty(1, 0, llm.get_input_embeddings().weight.shape[1],
                              device=llm.device, dtype=torch.bfloat16)
    audio_position = torch.tensor([0], device=llm.device)

    for index, question in enumerate(questions, 1):
        for condition in conditions:
            key = (question["question_id"], condition)
            if key in results:
                continue
            text = make_prompt(question["question"], condition)
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True,
                tokenize=True, return_tensors="pt",
            ).to(llm.device)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            print(f"[{index}/{len(questions)}] {question['question_id']} {condition}", flush=True)
            with torch.inference_mode():
                tokens, forwards = generate(
                    model, encoded, audio_info_lengths=audio_position, speech_embeds=empty_audio,
                    steps=128, gen_length=args.max_new_tokens, block_length=32,
                    temperature=0.0, factor=1.0,
                )
            response = tokenizer.decode(tokens[0, encoded.shape[1]:], skip_special_tokens=True)
            row = {**question, "condition": condition, "prompt": text, "response": response,
                   "model": MODEL, "seed": args.seed, "forwards": int(forwards),
                   "steps": 128, "block_length": 32, "max_new_tokens": args.max_new_tokens,
                   "temperature": 0.0, "decoder": "fast_dllm", "factor": 1.0,
                   **score_response(response, question["answer"])}
            with results_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            results[key] = row
            print(f"  correct={row['correct']} answer={row['predicted_answer']} forwards={forwards}", flush=True)
    summarize(results)


if __name__ == "__main__":
    main()
