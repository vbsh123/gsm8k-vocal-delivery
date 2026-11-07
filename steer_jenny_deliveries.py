"""Test fixed angry steering on requested Jenny deliveries, with per-delivery baselines.

No angry partner, correctness filter, or emotion-classifier agreement required.
First-batch testing must explicitly allow in-sample use; it is reported separately.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import wave

from cloud_tts_smoke import STYLES, AZURE_EXTRA_STYLES
from scoring import score_response
from steer import read_rows, summarize
from steer_jenny_complete import CONDITIONS
from steer_jenny_six import main as run_diagnostic


def select_recordings(paths, styles):
    recordings, seen = [], set()
    skipped = Counter()
    for path in paths:
        for task in read_rows(path/"plan.jsonl"):
            if task["provider"] != "azure" or task["style"] not in styles:
                continue
            key = (task["question_id"], task["voice"], task["style"])
            if key in seen:
                raise ValueError(f"Duplicate question/voice/delivery: {key}")
            seen.add(key)
            audio_path = (path/task["audio_path"]).resolve()
            with wave.open(str(audio_path), "rb") as handle:
                duration = handle.getnframes()/handle.getframerate()
            if not 0 < duration <= 30:
                skipped[task["style"]] += 1
                continue
            recordings.append(dict(clip_id=f"{key[0]}__{key[1]}__{key[2]}__r00",
                question_id=key[0], style=key[2], speaker=key[1], repeat=0,
                duration_seconds=duration, audio_path=str(audio_path),
                answer=str(task["answer"]).rsplit("####", 1)[-1].strip()))
    return sorted(recordings, key=lambda r: (r["question_id"], r["speaker"], r["style"])), skipped


def report(output):
    config = json.loads((output/"run_config.json").read_text())
    latest = {(r["clip_id"], r["condition"]): {**r, **score_response(r["response"], r["reference_answer"])}
              for r in read_rows(output/"results.jsonl")}
    conditions = config["settings"]["conditions"]
    ids = {r["clip_id"] for r in config["recordings"]}
    print(f"\nCompleted generations: {sum(cid in ids for cid, _ in latest)}/{len(ids)*len(conditions)}")
    if config["settings"].get("allow_extraction_overlap"):
        print("IN-SAMPLE: questions overlap the vector-extraction pool. Not held-out performance.")
    else:
        print("Questions do not overlap vector extraction; earlier tuning still makes this exploratory.")
    summaries = {}
    for style in sorted({r["style"] for r in config["recordings"]}):
        selected = {r["clip_id"] for r in config["recordings"] if r["style"] == style}
        rows = [row for (cid, _), row in latest.items() if cid in selected]
        print(f"\nInput delivery: {style}. Baseline = that SAME recording without steering.")
        summaries[style] = summarize(rows, conditions)
    return summaries


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", nargs="+", type=Path, default=[Path("outputs/azure-jenny500-neutral")])
    p.add_argument("--styles", nargs="+", choices=STYLES+AZURE_EXTRA_STYLES, default=["neutral"])
    p.add_argument("--vectors", type=Path, default=Path("outputs/jenny-angry-vectors/style-vectors.pt"))
    p.add_argument("--previous", type=Path, default=Path("outputs/steer-jenny-angry-L0-25-n50"))
    p.add_argument("--output", type=Path, default=Path("outputs/steer-jenny-first500-insample"))
    p.add_argument("--in-sample", action="store_true", help="Explicitly allow vector-extraction question overlap.")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args(argv)
    if args.summary_only:
        report(args.output)
        return
    if len(set(args.styles)) != len(args.styles):
        p.error("Styles must be unique")
    recordings, skipped = select_recordings(args.paths, args.styles)
    if not recordings:
        p.error("No eligible recordings")
    training = {r["question_id"] for r in read_rows(args.vectors.parent/"extraction-questions.jsonl")}
    overlap = training & {r["question_id"] for r in recordings}
    if overlap and not args.in_sample:
        p.error("These questions overlap vector extraction. Add --in-sample to label this explicitly.")
    prepared = dict(recordings=recordings, conditions=CONDITIONS,
                    selection="requested deliveries; no correctness, classifier, or partner selection",
                    in_sample=args.in_sample, extraction_overlap_question_ids=sorted(overlap))
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output/"selected-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != prepared:
        p.error("Saved selection changed; use a different --output")
    plan_path.write_text(json.dumps(prepared, indent=2)+"\n")
    print("Selected clips:", dict(Counter(r["style"] for r in recordings)), flush=True)
    print("Skipped invalid/over-30-second clips:", dict(skipped), flush=True)
    command = ["--previous", str(args.previous), "--vectors", str(args.vectors),
               "--output", str(args.output), "--prepared-plan", str(plan_path)]
    if args.in_sample:
        command.append("--allow-extraction-overlap")
    if args.prepare_only:
        command.append("--prepare-only")
    run_diagnostic(command)
    if not args.prepare_only:
        summary = report(args.output)
        (args.output/"summary-by-delivery.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
