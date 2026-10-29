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
    prompt = (
        "Write a natural spoken request for help with the math question below. "
        f"The speaker's attitude toward the listener is {args.style}. "
        "Start with exactly one short remark, at most twelve words, that subtly "
        "conveys that attitude through everyday wording. Make the remark relevant "
        "to asking the listener for help with this problem. "
        "Avoid explicitly naming the emotion, theatrical language, shouting, "
        "and threats. Choose the wording yourself. "
        "After the remark, copy the entire original question exactly, "
        "preserving every word and number, including the final question. "
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
