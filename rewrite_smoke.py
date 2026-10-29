"""Ask LLaDA to add style-appropriate remarks to one question, then save TTS input."""

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

# The upstream text decoder imports `model` relative to its own directory.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "external/DIFFA/DIFFA-2/src/fast_dllm"))
from model.modeling_llada import LLaDAModelLM
from origin_generate import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=ROOT / "examples/tiny.jsonl")
    parser.add_argument("--style", default="angry")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/wording-smoke")
    args = parser.parse_args()

    question = json.loads(args.input_jsonl.read_text().splitlines()[0])
    styles = json.loads((ROOT / "configs/styles.json").read_text())
    prompt = (
        f"Write a spoken script expressing this style toward the listener: {args.style}. "
        f"Style description: {styles[args.style]}\n"
        f"You MUST add at least one new remark that expresses {args.style} "
        "directly toward the listener, using emotionally expressive words. "
        "Returning the original question unchanged is not acceptable. "
        "Choose the remark yourself and put it before the math question. "
        "Then include the entire original question, preserving every word and number. "
        "Do not solve the problem, suggest an answer, or add mathematical facts. "
        "Output only the complete script, without explanations or quotation marks.\n\n"
        "Original question:\n"
        + question["question"]
    )

    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = LLaDAModelLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda().eval()
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(chat, return_tensors="pt").input_ids.cuda()
    with torch.inference_mode():
        tokens, _ = generate(
            model, inputs, steps=128, gen_length=256, block_length=32,
            temperature=0.0,
        )
    answer_tokens = tokens[0, inputs.shape[1]:].tolist()
    # Decode only through the first end-of-turn token, not the unused canvas.
    end = (answer_tokens + [tokenizer.eos_token_id]).index(tokenizer.eos_token_id)
    rewritten = tokenizer.decode(answer_tokens[:end], skip_special_tokens=True).strip()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rewrite.json").write_text(json.dumps({
        "original": question["question"], "style": args.style,
        "prompt": prompt, "rewritten": rewritten,
    }, indent=2) + "\n")
    (args.output / "tts-input.jsonl").write_text(
        json.dumps(dict(question, question=rewritten)) + "\n"
    )
    print(rewritten, flush=True)


if __name__ == "__main__":
    main()
