"""Compare saved Jenny delivery accuracy on matched questions, with and without emotion filtering."""

import argparse
from pathlib import Path

from compare_cloud_accuracy import read, summarize_deliveries
from cloud_tts_smoke import STYLES, AZURE_EXTRA_STYLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, nargs="+", default=[
        Path("outputs/azure-jenny500-extra"), Path("outputs/azure-jenny500-neutral")])
    parser.add_argument("--classifiers", nargs="+", choices=["emotion2vec", "sensevoice"],
                        default=["emotion2vec", "sensevoice"])
    parser.add_argument("--styles", nargs="+", choices=STYLES + AZURE_EXTRA_STYLES,
                        default=["neutral", "angry", "happy", "sad"])
    parser.add_argument("--results-file", default="results-matched-accuracy.jsonl",
                        help="Answer JSONL filename inside each input folder; defaults to DIFFA results.")
    args = parser.parse_args()
    tasks, results = [], {}
    labels = {name: {} for name in args.classifiers}
    for folder in args.paths:
        prefix = str(folder.resolve()) + "/"
        tasks.extend({**row, "audio_path": prefix + row["audio_path"]}
                     for row in read(folder / "plan.jsonl") if row["provider"] == "azure")
        result_path = folder / args.results_file
        if result_path.exists():
            results.update({prefix + row["audio_path"]: row for row in read(result_path)})
        for name in args.classifiers:
            label_path = folder / f"results-{name}.jsonl"
            if label_path.exists():
                labels[name].update({prefix + row["audio_path"]: row for row in read(label_path)})
    styles = args.styles
    print(f"Accuracy results: {args.results_file}")
    print("\nClassifier agreement with intended style (all generated clips)")
    print(f'{"Model":12} {"Style":10} {"Classified":>11} {"Matching":>10} {"Match rate":>11} {"Missing":>9}')
    for name, predictions in labels.items():
        for style in styles:
            intended = [t for t in tasks if t["style"] == style]
            selected = [predictions[t["audio_path"]] for t in intended if t["audio_path"] in predictions]
            n = len(selected)
            k = sum(row["emotion"] == style for row in selected)
            supported = style in STYLES
            rate = f"{k/n:.1%}" if n and supported else "n/a"
            matching = str(k) if supported else "n/a"
            print(f'{name:12} {style:10} {n:11} {matching:>10} {rate:>11} {len(intended)-n:9}')
            if not supported and selected:
                from collections import Counter
                print("  Predicted labels:", dict(Counter(row["emotion"] for row in selected)))

    print("\nALL EVALUATED CLIPS — PRIMARY; PAIRED ACCURACY ONLY")
    summarize_deliveries(tasks, results, styles)
    for name, predictions in labels.items():
        if predictions:
            summarize_deliveries(tasks, results, [s for s in styles if s in STYLES], predictions, name)
    if len(labels) > 1 and all(labels.values()):
        any_agreed = {t["audio_path"]: {"emotion": t["style"]} for t in tasks
                      if any(predictions.get(t["audio_path"], {}).get("emotion") == t["style"]
                             for predictions in labels.values())}
        summarize_deliveries(tasks, results, [s for s in styles if s in STYLES],
                             any_agreed, "at least one classifier")
        agreed = {t["audio_path"]: {"emotion": t["style"]} for t in tasks
                  if all(predictions.get(t["audio_path"], {}).get("emotion") == t["style"]
                         for predictions in labels.values())}
        summarize_deliveries(tasks, results, [s for s in styles if s in STYLES], agreed, "+".join(labels))
    print("Filtered comparisons require the requested emotion on both sides, including neutral.")
    if len(labels) > 1:
        print("At least one: each clip needs a matching classifier; the classifier may differ between the two clips.")
    if any(s not in STYLES for s in styles):
        print("Additional Azure styles have no exact classifier label; their comparisons are unfiltered only.")


if __name__ == "__main__":
    main()
