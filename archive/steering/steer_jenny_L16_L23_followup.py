"""L16/L23 answer: recovery at x0.5/x1, then regressions at x1.25.

Seed 42 before every generation. Reuse only matching seed42 baselines; never
reuse old per-clip-derived-seed answers. Stages run in separate GPU processes.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from compare_cloud_accuracy import PROMPT
from scoring import score_response
from steer import read_rows, summarize
from archive.steering.steer_jenny_recovery import recovery_summary
from archive.steering.steer_jenny_regressions import regression_summary
from archive.steering.steer_jenny_recovery74_promoted import report_unseen
from archive.steering.steer_jenny_six import main as run


def conditions(stage):
    strengths=[0.5,1] if stage=="recovery" else [1.25]
    return [dict(name="baseline",interventions=[])]+[
        dict(name=f"L{layer}-answer-a{a:g}",interventions=[dict(layer=layer,group="answer",strength=a)])
        for layer in [16,23] for a in strengths]


def copy_baselines(source,output,records):
    previous={r["clip_id"]:r for r in read_rows(source/"results.jsonl") if r["condition"]=="baseline"}
    path=output/"results.jsonl"
    saved={(r["clip_id"],r["condition"]):r for r in read_rows(path)}
    copied=0
    with path.open("a") as stream:
        for record in records:
            row=previous.get(record["clip_id"])
            if row is None or row.get("seed")!=42:
                continue
            if (row.get("prompt")!=PROMPT or row.get("max_new_tokens")!=512
                    or row.get("decoder")!="fast_dllm" or row.get("steered_forwards")!=0
                    or row.get("interventions")!=[] or row.get("question_id")!=record["question_id"]
                    or str(row.get("reference_answer"))!=str(record["answer"])):
                continue
            score=score_response(row["response"],record["answer"])
            if any(row.get(k)!=v for k,v in score.items()):
                raise ValueError(f"Baseline score differs at {record['clip_id']}")
            key=(row["clip_id"],"baseline")
            if key in saved:
                if saved[key]!=row:
                    raise ValueError(f"Existing baseline conflicts at {record['clip_id']}")
                continue
            stream.write(json.dumps(row)+"\n")
            stream.flush()
            copied+=1
    return copied


def report(output,stage):
    plan=json.loads((output/"selected-plan.json").read_text())
    if stage=="recovery":
        recovery_summary(output)
        report_unseen(output,set(plan["screen_question_ids"]))
    else:
        regression_summary(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["all","recovery","regressions"],default="all")
    p.add_argument("--recovery-source",type=Path,default=Path("outputs/steer-jenny-recovery74-promoted-a1.25-seed42"))
    p.add_argument("--regression-source",type=Path,default=Path("outputs/steer-jenny-angry-regressions"))
    p.add_argument("--vectors",type=Path,default=Path("outputs/jenny-angry-vectors/style-vectors.pt"))
    p.add_argument("--output",type=Path,default=Path("outputs/steer-jenny-L16-L23-followup-seed42"))
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--summary-only",action="store_true")
    args=p.parse_args()
    if args.stage=="all":
        if not args.summary_only and not args.prepare_only:
            # Check both stages before starting a long GPU run.
            for stage in ["recovery","regressions"]:
                subprocess.run([sys.executable,"-u",str(Path(__file__).resolve()),
                                *sys.argv[1:],"--stage",stage,"--prepare-only"],check=True)
        for stage in ["recovery","regressions"]:
            command=[sys.executable,"-u",str(Path(__file__).resolve()),
                     *sys.argv[1:],"--stage",stage]
            subprocess.run(command,check=True)
        return
    output=args.output/args.stage
    if args.summary_only:
        report(output,args.stage)
        return
    source=args.recovery_source if args.stage=="recovery" else args.regression_source
    config=json.loads((source/"run_config.json").read_text())
    records=config["recordings"]
    expected=74 if args.stage=="recovery" else 260
    if len(records)!=expected or len({r["clip_id"] for r in records})!=expected:
        p.error(f"Expected the same {expected}-recording source cohort")
    stat=args.vectors.stat()
    old=config["settings"]
    if (args.vectors.resolve()!=Path(old["vectors"]).resolve()
            or stat.st_size!=old["vector_bytes"] or stat.st_mtime_ns!=old["vector_mtime_ns"]):
        p.error("Vector differs from the source run")
    if any(r["style"]!="neutral" or not Path(r["audio_path"]).is_absolute()
           or not Path(r["audio_path"]).is_file() for r in records):
        p.error("Expected original neutral recordings at their saved absolute paths")
    screen=[]
    if args.stage=="recovery":
        screen=json.loads((source/"selected-plan.json").read_text())["screen_question_ids"]
        if len(set(screen))!=10 or not set(screen)<={r["question_id"] for r in records}:
            p.error("Invalid ten-question screening cohort")
    plan=dict(recordings=records,conditions=conditions(args.stage),screen_question_ids=screen,
              selection=f"Same {expected} recordings from {source.name}; {args.stage} follow-up",
              source_run=str(source.resolve()),seed=42)
    output.mkdir(parents=True,exist_ok=True)
    path=output/"selected-plan.json"
    if path.exists() and json.loads(path.read_text())!=plan:
        p.error("Frozen plan changed; use another --output")
    path.write_text(json.dumps(plan,indent=2)+"\n")
    command=["--previous",str(source),"--vectors",str(args.vectors),
             "--prepared-plan",str(path),"--output",str(output)]
    # Validate/freeze settings before importing any compatible cached answers.
    run(command+["--prepare-only"])
    copied=copy_baselines(source,output,records)
    print(f"{args.stage}: copied {copied} compatible seed42 baselines. Other baselines regenerate at seed42.",flush=True)
    if args.prepare_only:
        return
    with patch("archive.steering.steer_jenny_six.report",side_effect=lambda rows,cs,rs:summarize(rows,cs)):
        run(command)
    report(output,args.stage)


if __name__=="__main__":
    main()
