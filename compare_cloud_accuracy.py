"""Evaluate Azure clips against saved older TTS results on the same question/style."""

import argparse
from collections import Counter
from itertools import combinations
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


def summarize_all(tasks, results, older, styles):
    latest = {r["clip_id"]: r for r in older}
    print("\nOverall GSM8K accuracy (strict scoring; different question sets)")
    print(f'{"Style":10} {"Azure correct/N":>17} {"Azure acc":>11} '
          f'{"Older correct/N":>17} {"Older acc":>11}')
    for style in styles:
        new = [results[t["audio_path"]] for t in tasks
               if t["style"] == style and t["audio_path"] in results]
        old = [r for r in latest.values() if r["style"] == style
               and r["speaker"] == "Ryan" and r["repeat"] == 0]
        values = []
        for rows in (new, old):
            k = sum(score_response(r["response"], r["reference_answer"])["correct"] for r in rows)
            values.extend([f"{k}/{len(rows)}", f"{k/len(rows):.1%}" if rows else "n/a"])
        print(f'{style:10} {values[0]:>17} {values[1]:>11} {values[2]:>17} {values[3]:>11}')
    print("All available clips per style; this is an unpaired descriptive comparison.")


def summarize_deliveries(tasks, results, styles, labels=None):
    groups = {style: {} for style in styles}
    for task in tasks:
        if task["style"] not in groups or task["audio_path"] not in results:
            continue
        if labels is not None and labels.get(task["audio_path"], {}).get("emotion") != task["style"]:
            continue
        row = results[task["audio_path"]]
        key = (task["question_id"], task["voice"])
        groups[task["style"]][key] = score_response(row["response"], row["reference_answer"])["correct"]
    subset = "emotion2vec-agreed on both sides" if labels is not None else "all requested deliveries"
    print(f"\nWithin-Azure, same question and voice ({subset})")
    print(f'{"Reference":10} {"Test":10} {"Pairs":>6} {"Ref acc":>9} {"Test acc":>9} '
          f'{"Gain pp":>9} {"Wins":>6} {"Losses":>7} {"Exact p":>9}')
    for reference, test in combinations(styles, 2):
        keys = groups[reference].keys() & groups[test].keys()
        n = len(keys)
        if not n:
            continue
        a, b = groups[reference], groups[test]
        wins = sum(b[k] and not a[k] for k in keys)
        losses = sum(a[k] and not b[k] for k in keys)
        d = wins + losses
        p = min(1.0, 2 * sum(comb(d, k) for k in range(min(wins, losses) + 1)) / 2**d)
        ref_acc, test_acc = sum(a[k] for k in keys)/n, sum(b[k] for k in keys)/n
        print(f'{reference:10} {test:10} {n:6} {ref_acc:9.1%} {test_acc:9.1%} '
              f'{100*(test_acc-ref_acc):+9.1f} {wins:6} {losses:7} {p:9.4f}')
    print("Gain and wins/losses refer to test versus reference. McNemar p-values are unadjusted.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("outputs/azure-jenny50"))
    parser.add_argument("--older-results", type=Path,
                        default=Path("outputs/manual200-mixed/cropped/results-reasoning-v2.jsonl"))
    parser.add_argument("--styles", nargs="+", choices=["angry", "happy", "sad"], default=["angry", "happy"])
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--all-cloud", action="store_true", help="Evaluate all selected Azure clips, including unmatched questions.")
    parser.add_argument("--older-emotion-agreed", action="store_true",
                        help="Keep older clips only when emotion2vec agrees with their requested style.")
    parser.add_argument("--older-labels", type=Path,
                        default=Path("outputs/manual200-mixed/cropped/results-emotion2vec-evaluated.jsonl"))
    args = parser.parse_args()
    tasks = read(args.path / "plan.jsonl")
    older = read(args.older_results)
    if args.older_emotion_agreed:
        labels = {r["clip_id"]: r for r in read(args.older_labels)}
        older = [r for r in older if labels.get(r["clip_id"], {}).get("emotion") == r["style"]]
        print("Older TTS subset: requested style equals emotion2vec prediction.", flush=True)
    pairs = matched_tasks(tasks, older, args.styles)
    counts = Counter(task["style"] for task, old in pairs)
    for style in args.styles:
        total = sum(t["provider"] == "azure" and t["style"] == style for t in tasks)
        print(f"{style}: {counts[style]}/{total} Azure clips have an evaluated older Ryan/repeat-0 partner.")
    if args.all_cloud:
        references = {task["audio_path"]: old for task, old in pairs}
        pairs = [(task, references.get(task["audio_path"])) for task in tasks
                 if task["provider"] == "azure" and task["style"] in args.styles]
    output = args.path / "results-matched-accuracy.jsonl"
    results = {r["audio_path"]: r for r in read(output)} if output.exists() else {}
    def report(selected):
        if args.all_cloud:
            selected_tasks = [task for task, old in selected]
            summarize_all(selected_tasks, results, older, args.styles)
            summarize_deliveries(selected_tasks, results, args.styles)
            labels_path = args.path / "results-emotion2vec.jsonl"
            if labels_path.exists():
                labels = {r["audio_path"]: r for r in read(labels_path)}
                summarize_deliveries(selected_tasks, results, args.styles, labels)
        else:
            summarize(selected, results, args.styles)
    if args.prepare_only or args.summary_only:
        report(pairs)
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
                       "older_clip_id": old["clip_id"] if old else None, "response": response,
                       "prompt": PROMPT, "max_new_tokens": 512, "decoder": "fast_dllm",
                       "seed": 42, **score}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                results[task["audio_path"]] = row
                print(f'[{index}/{len(pending)}] {task["question_id"]} {task["style"]}: '
                      f'correct={score["correct"]} answer={score["predicted_answer"]}', flush=True)
    report(eligible)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
