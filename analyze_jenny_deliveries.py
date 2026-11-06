"""Compare saved Jenny delivery accuracy on matched questions, with and without emotion filtering."""

import argparse
from pathlib import Path

from compare_cloud_accuracy import read, summarize_deliveries
from scoring import score_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, nargs="+", default=[
        Path("outputs/azure-jenny500-extra"), Path("outputs/azure-jenny500-neutral")])
    args = parser.parse_args()
    tasks, results, labels = [], {}, {}
    for folder in args.paths:
        prefix = str(folder.resolve()) + "/"
        tasks.extend({**row, "audio_path": prefix + row["audio_path"]}
                     for row in read(folder / "plan.jsonl") if row["provider"] == "azure")
        result_path = folder / "results-matched-accuracy.jsonl"
        if result_path.exists():
            results.update({prefix + row["audio_path"]: row for row in read(result_path)})
        label_path = folder / "results-emotion2vec.jsonl"
        if label_path.exists():
            labels.update({prefix + row["audio_path"]: row for row in read(label_path)})
    styles = ["neutral", "angry", "happy", "sad"]
    for filtered in (False, True):
        print("\n" + ("EMOTION2VEC-AGREED CLIPS" if filtered else "ALL EVALUATED CLIPS — PRIMARY"))
        print(f'{"Style":10} {"Correct/N":>12} {"Accuracy":>10}')
        for style in styles:
            selected = [results[t["audio_path"]] for t in tasks if t["style"] == style
                        and t["audio_path"] in results
                        and (not filtered or labels.get(t["audio_path"], {}).get("emotion") == style)]
            n = len(selected)
            k = sum(score_response(row["response"], row["reference_answer"])["correct"] for row in selected)
            accuracy = f"{k/n:.1%}" if n else "n/a"
            print(f'{style:10} {f"{k}/{n}":>12} {accuracy:>10}')
        summarize_deliveries(tasks, results, styles, labels if filtered else None)
    print("Filtered comparisons require the requested emotion on both sides, including neutral.")


if __name__ == "__main__":
    main()
