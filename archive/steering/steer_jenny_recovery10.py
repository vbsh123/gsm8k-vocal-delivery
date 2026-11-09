"""Screen new layer/group pairs at x1.25 on ten recovery74 questions."""

import argparse
import hashlib
import json
from pathlib import Path
import random
from unittest.mock import patch

from steer import read_rows, summarize
from archive.steering.steer_jenny_recovery import recovery_summary
from archive.steering.steer_jenny_six import main as run


def prepare(source, pilot):
    records=source["recordings"]
    pilot_ids={r["question_id"] for r in pilot["recordings"]}
    if len(records)!=74 or len({r["question_id"] for r in records})!=74:
        raise ValueError("Expected the 74-question recovery cohort")
    if any(r["question_id"] in pilot_ids or r["style"]!="neutral" for r in records):
        raise ValueError("Recovery cohort includes an initial-pilot question or non-neutral input")
    tested=set()
    for c in source["settings"]["conditions"]:
        if c["name"]!="baseline":
            tested.update((t["layer"],t["group"]) for t in c.get("interventions",[c]))
    targets=[(l,g) for l in range(32) for g in ["acoustic","answer"] if (l,g) not in tested]
    if not targets:
        raise ValueError("All requested layer/group pairs were tested")
    selected=sorted(records,key=lambda r:r["question_id"])
    random.Random(42).shuffle(selected)
    conditions=[dict(name="baseline",interventions=[])]
    for l,g in targets:
        conditions.append(dict(name=f"L{l}-{g}-a1.25",interventions=[dict(layer=l,group=g,strength=1.25)]))
    return selected[:10],conditions,tested


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run",type=Path,default=Path("outputs/steer-jenny-angry-recovery74-expanded"))
    p.add_argument("--pilot-run",type=Path,default=Path("outputs/steer-jenny-angry-L0-25-n50"))
    p.add_argument("--vectors",type=Path,default=Path("outputs/jenny-angry-vectors/style-vectors.pt"))
    p.add_argument("--output",type=Path,default=Path("outputs/steer-jenny-recovery10-newlayers-a1.25-seed42"))
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--summary-only",action="store_true")
    args=p.parse_args()
    if args.summary_only:
        recovery_summary(args.output)
        return
    source_path=args.source_run/"run_config.json"
    source=json.loads(source_path.read_text())
    pilot=json.loads((args.pilot_run/"run_config.json").read_text())
    try:
        selected,conditions,tested=prepare(source,pilot)
    except ValueError as exc:
        p.error(str(exc))
    stat=args.vectors.stat()
    old=source["settings"]
    if (args.vectors.resolve()!=Path(old["vectors"]).resolve()
            or stat.st_size!=old["vector_bytes"] or stat.st_mtime_ns!=old["vector_mtime_ns"]):
        p.error("Vector differs from the 74-question recovery run")
    done={(r["clip_id"],r["condition"]) for r in read_rows(args.source_run/"results.jsonl")}
    if any((r["clip_id"],c["name"]) not in done for r in source["recordings"]
           for c in source["settings"]["conditions"]):
        p.error("Source 74-question run is incomplete")
    training={r["question_id"] for r in read_rows(args.vectors.parent/"extraction-questions.jsonl")}
    if training & {r["question_id"] for r in selected}:
        p.error("Selected questions overlap extraction")
    if any(not Path(r["audio_path"]).is_absolute() or not Path(r["audio_path"]).is_file() for r in selected):
        p.error("Selected recordings need existing absolute audio paths")
    plan=dict(recordings=selected,conditions=conditions,
              selection="Ten second-batch angry-correct/neutral-wrong recovery74 questions; initial pilot excluded",
              excluded_layer_groups=[dict(layer=l,group=g) for l,g in sorted(tested)],
              source_run=str(args.source_run.resolve()),
              source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),seed=42)
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/"selected-plan.json"
    if path.exists() and json.loads(path.read_text())!=plan:
        p.error("Frozen plan changed; use another output folder")
    path.write_text(json.dumps(plan,indent=2)+"\n")
    print(f"Excluded {len(tested)} previously tested layer/group pairs, regardless of strength.",flush=True)
    print(f"10 questions x {len(conditions)-1} new targets + baselines = {10*len(conditions)} generations.",flush=True)
    print("Strength 1.25; acoustic/answer separately; all 32 layers considered; seed always 42.",flush=True)
    print("Questions:",", ".join(r["question_id"] for r in selected),flush=True)
    command=["--previous",str(args.source_run),"--vectors",str(args.vectors),
             "--prepared-plan",str(path),"--output",str(args.output)]
    if args.prepare_only:
        command.append("--prepare-only")
    with patch("archive.steering.steer_jenny_six.report",side_effect=lambda rows,cs,rs:summarize(rows,cs)):
        run(command)
    if not args.prepare_only:
        recovery_summary(args.output)


if __name__=="__main__":
    main()
