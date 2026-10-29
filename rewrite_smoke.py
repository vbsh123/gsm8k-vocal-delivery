"""Ask Qwen to add style-appropriate remarks to one question, then save TTS input."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent


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
        "Start with exactly one short remark, at most twelve words, that clearly "
        "conveys that attitude through everyday wording. Make the remark relevant "
        "to asking the listener for help with this problem. "
        "For angry, sound clearly irritated and impatient with the listener, "
        "as if they have repeatedly failed to help. Keep it conversational. "
        "Mere surprise, disbelief, or a sarcastic question is not enough. "
        "Avoid explicitly naming the emotion, theatrical language, shouting, "
        "and threats. Choose the wording yourself. "
        "After the remark, copy the entire original question exactly, "
        "preserving every word and number, including the final question. "
        "Do not solve the problem, suggest an answer, or add mathematical facts. "
        "Output only the complete script, without explanations or quotation marks.\n\n"
        "Original question:\n"
        + question["question"]
    )

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
    ).cuda().eval()
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(chat, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        tokens = model.generate(
            **inputs, max_new_tokens=256, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    rewritten = tokenizer.decode(
        tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True,
    ).strip()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rewrite.json").write_text(json.dumps({
        "model": model_id, "original": question["question"], "style": args.style,
        "prompt": prompt, "rewritten": rewritten,
    }, indent=2) + "\n")
    (args.output / "tts-input.jsonl").write_text(
        json.dumps(dict(question, question=rewritten)) + "\n"
    )
    print(rewritten, flush=True)


if __name__ == "__main__":
    main()
