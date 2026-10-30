#!/usr/bin/env python3
"""Replace one delivery in a generated dataset, backing up files and metadata."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from generate import completed_records, render_preview, write_json, write_jsonl


def replace_style(target, source, style):
    target, source = Path(target).resolve(), Path(source).resolve()
    if target == source:
        raise ValueError("Source and target must be different directories.")
    target_config = json.loads((target / "run_config.json").read_text())
    source_config = json.loads((source / "run_config.json").read_text())
    tasks = target_config["tasks"]
    old_tasks = {t["clip_id"]: t for t in tasks if t["style"] == style}
    new_tasks = {t["clip_id"]: t for t in source_config["tasks"] if t["style"] == style}
    if not old_tasks or old_tasks.keys() != new_tasks.keys():
        raise ValueError("Replacement clips must match all target clip IDs for this style. "
                         "Use the same question count, seed, voices, and repeats.")
    for key in ("model", "model_revision", "temperature", "max_new_tokens", "language", "wav_subtype"):
        if target_config[key] != source_config[key]:
            raise ValueError(f"Generation setting differs: {key}")
    records = completed_records(target)
    replacements = completed_records(source)
    for clip_id, old in old_tasks.items():
        new = new_tasks[clip_id]
        for key in ("question_id", "text", "answer", "speaker", "repeat", "seed", "audio_path"):
            if old[key] != new[key]:
                raise ValueError(f"Mismatched {key} for {clip_id}")
        wav = source / new["audio_path"]
        if clip_id not in replacements or not wav.is_file() or wav.stat().st_size <= 44:
            raise ValueError(f"Replacement is not complete: {clip_id}")
        record = replacements[clip_id]
        if any(record.get(k) != v for k, v in new.items()):
            raise ValueError(f"Replacement manifest disagrees with its plan: {clip_id}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = target / "backups" / f"{style}-{stamp}"
    backup.mkdir(parents=True)
    for name in ("run_config.json", "plan.jsonl", "manifest.jsonl", "listen.html", "results.jsonl"):
        if (target / name).exists():
            shutil.copy2(target / name, backup / name)
    for old in old_tasks.values():
        wav = target / old["audio_path"]
        if wav.exists():
            saved = backup / old["audio_path"]
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav, saved)

    for clip_id, new in new_tasks.items():
        wav = target / new["audio_path"]
        wav.parent.mkdir(parents=True, exist_ok=True)
        temporary = wav.with_suffix(".replacement.wav")
        shutil.copy2(source / new["audio_path"], temporary)
        temporary.replace(wav)
        records[clip_id] = replacements[clip_id]
    tasks = [new_tasks.get(t["clip_id"], t) for t in tasks]
    target_config["tasks"] = tasks
    # Other styles may have come from the original environment; retain both.
    if (source / "environment.json").exists():
        shutil.copy2(source / "environment.json", target / f"environment-{style}-{stamp}.json")
    write_json(target / "run_config.json", target_config)
    write_jsonl(target / "plan.jsonl", tasks)
    write_jsonl(target / "manifest.jsonl", [records[t["clip_id"]] for t in tasks if t["clip_id"] in records])
    render_preview(target, tasks, records)
    # Existing scores refer to a different set of waveforms. Already backed up.
    if (target / "results.jsonl").exists():
        (target / "results.jsonl").unlink()
        print("Archived existing results.jsonl in the backup; rerun evaluation for the updated dataset.")
    print(f"Replaced {len(new_tasks)} {style} clips in {target}")
    print(f"Backup: {backup}")
    return backup


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--style", required=True)
    args = p.parse_args()
    try:
        replace_style(args.target, args.source, args.style)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
