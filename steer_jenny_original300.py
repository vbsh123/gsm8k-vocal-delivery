"""Transfer Azure angry directions to 300 original Spoken GSM recordings.

No emotion/correctness filter. Exactly seed 42 before baseline and each steering
condition. Original audio IDs cannot prove question disjointness from extraction.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from steer import read_rows, summarize
from steer_jenny_six import main as run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path",type=Path,default=Path("outputs/spoken-gsm-original"))
    p.add_argument("--vectors",type=Path,default=Path("outputs/jenny-angry-vectors/style-vectors.pt"))
    p.add_argument("--previous",type=Path,default=Path("outputs/steer-jenny-angry-complete"))
    p.add_argument("--output",type=Path,default=Path("outputs/steer-jenny-original300-seed42"))
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--summary-only",action="store_true")
    args=p.parse_args()
    if args.summary_only:
        config=json.loads((args.output/"run_config.json").read_text())
        summarize(read_rows(args.output/"results.jsonl"),config["settings"]["conditions"])
        return
    if not args.vectors.is_file():
        p.error(f"Missing vector checkpoint: {args.vectors}")
    records=read_rows(args.path/"manifest.jsonl")
    if len({r["clip_id"] for r in records})!=len(records):
        p.error("Duplicate original manifest IDs")
    selected=[r for r in records if r["style"]=="original" and 0<r["duration_seconds"]<=30][:300]
    if len(selected)!=300:
        p.error("Prepare 300 eligible originals with prepare_spoken_gsm.py --num-examples 300")
    selected=[{**r,"audio_path":str((args.path/r["audio_path"]).resolve())} for r in selected]
    if any(not Path(r["audio_path"]).is_file() for r in selected):
        p.error("Missing original audio files")
    conditions=[dict(name="baseline",interventions=[])]
    for layer,group in [(11,"answer"),(12,"acoustic")]:
        conditions.append(dict(name=f"L{layer}-{group}-a1",interventions=[dict(layer=layer,group=group,strength=1)]))
    plan=dict(recordings=selected,conditions=conditions,
              selection="First 300 eligible original recordings; no emotion or correctness filtering",
              question_overlap="Unverified across original audio and generated text-question IDs",
              seed_policy="constant 42 before every generation")
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/"selected-plan.json"
    if path.exists() and json.loads(path.read_text())!=plan:
        p.error("Frozen cohort changed; use a new output folder")
    path.write_text(json.dumps(plan,indent=2)+"\n")
    print("300 originals × baseline / L11 answer x1 / L12 acoustic x1 = 900 generations.",flush=True)
    print("Exactly seed 42 before every generation. No old baseline answers imported.",flush=True)
    print("Audio-source transfer; question overlap with extraction is not verified.",flush=True)
    command=["--previous",str(args.previous),"--vectors",str(args.vectors),
             "--output",str(args.output),"--prepared-plan",str(path),"--fixed-seed","42"]
    if args.prepare_only:
        command.append("--prepare-only")
    # Reuse the inference runner, but not its outcome-selected six-question report.
    with patch("steer_jenny_six.report",side_effect=lambda rows,cs,records:summarize(rows,cs)):
        run(command)


if __name__=="__main__":
    main()
