"""Evaluate Azure clips against saved older TTS results on the same question/style."""

import argparse
from collections import Counter
import json
from math import comb
from pathlib import Path

from scoring import score_response


PROMPT = (
    "Solve the math problem in the audio. Give concise step-by-step reasoning "
    "with only the necessary calculations. If the question asks for a total, "
    "compute that total. End with one plain-text line in exactly this format: "
    "Final answer: <number>. Put a single numeric answer on that line, "
    "without Markdown, units, or separate component counts."
)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def matched_tasks(tasks, older, styles):
    latest = {r["clip_id"]: r for r in older}
    reference = {(r["question_id"], r["style"]): r for r in latest.values()
                 if r["speaker"] == "Ryan" and r["repeat"] == 0}
    return [(task, reference[task["question_id"], task["style"]]) for task in tasks
            if task["provider"] == "azure" and task["style"] in styles
            and (task["question_id"], task["style"]) in reference]


def summarize(pairs, results, styles):
    print("\nMatched GSM8K accuracy (strict scoring)")
    print(f'{"Style":10} {"Pairs":>6} {"Azure":>10} {"Older TTS":>10} {"Gain pp":>9} '
          f'{"Wins":>6} {"Losses":>7} {"Exact p":>9}')
    for style in styles:
        evaluated = [(results[task["audio_path"]], old) for task, old in pairs
                     if task["style"] == style and task["audio_path"] in results]
        if not evaluated:
            continue
        a = [score_response(new["response"], new["reference_answer"])["correct"] for new, old in evaluated]
        b = [score_response(old["response"], new["reference_answer"])["correct"] for new, old in evaluated]
        n = len(a)
        wins = sum(x and not y for x, y in zip(a, b))
        losses = sum(y and not x for x, y in zip(a, b))
        d = wins + losses
        p = min(1.0, 2 * sum(comb(d, k) for k in range(min(wins, losses) + 1)) / 2**d)
        print(f'{style:10} {n:6} {sum(a)/n:10.1%} {sum(b)/n:10.1%} '
              f'{100*(sum(a)-sum(b))/n:+9.1f} {wins:6} {losses:7} {p:9.4f}')
    print("Wins/losses refer to Azure versus older TTS. Parse failures count as incorrect.")
    print("Exact p is paired McNemar, unadjusted. Questions are matched; voices differ.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("outputs/azure-jenny50"))
    parser.add_argument("--older-results", type=Path,
                        default=Path("outputs/manual200-mixed/cropped/results-reasoning-v2.jsonl"))
    parser.add_argument("--styles", nargs="+", choices=["angry", "happy", "sad"], default=["angry", "happy"])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    tasks = read(args.path / "plan.jsonl")
    pairs = matched_tasks(tasks, read(args.older_results), args.styles)
    counts = Counter(task["style"] for task, old in pairs)
    for style in args.styles:
        total = sum(t["provider"] == "azure" and t["style"] == style for t in tasks)
        print(f"{style}: {counts[style]}/{total} Azure clips have an evaluated older Ryan/repeat-0 partner.")
    output = args.path / "results-matched-accuracy.jsonl"
    results = {r["audio_path"]: r for r in read(output)} if output.exists() else {}
    if args.prepare_only or args.summary_only:
        summarize(pairs, results, args.styles)
        return

    import soundfile as sf

    eligible = []
    for task, old in pairs:
        info = sf.info(args.path / task["audio_path"])
        if info.frames / info.samplerate <= 30:
            eligible.append((task, old))
        else:
            print(f'Skip over 30 seconds: {task["audio_path"]}', flush=True)
    pending = [(task, old) for task, old in eligible if task["audio_path"] not in results]
    print(f"{len(pending)} Azure clips remain to evaluate; saved results are skipped.", flush=True)
    if pending:
        import random
        import numpy as np
        import torch
        from extract_vectors import gen, load_model

        model, tokenizer, feature_extractor = load_model()
        with output.open("a") as stream:
            for index, (task, old) in enumerate(pending, 1):
                random.seed(42)
                np.random.seed(42)
                torch.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                audio, sample_rate = sf.read(args.path / task["audio_path"], dtype="float32")
                response = gen(audio, sample_rate, model, tokenizer, feature_extractor, PROMPT, 512)
                answer = str(task["answer"]).rsplit("####", 1)[-1].strip()
                score = score_response(response, answer)
                row = {"question_id": task["question_id"], "style": task["style"],
                       "voice": task["voice"], "audio_path": task["audio_path"],
                       "older_clip_id": old["clip_id"], "response": response,
                       "prompt": PROMPT, "max_new_tokens": 512, "decoder": "fast_dllm",
                       "seed": 42, **score}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                results[task["audio_path"]] = row
                print(f'[{index}/{len(pending)}] {task["question_id"]} {task["style"]}: '
                      f'correct={score["correct"]} answer={score["predicted_answer"]}', flush=True)
    summarize(eligible, results, args.styles)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
