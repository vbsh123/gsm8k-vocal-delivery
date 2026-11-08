"""Compare L12 acoustic x1 with L16 answer x1.25 on the same full460 cohort.

Seed42 throughout. Reuse only compatible saved seed42 answers; old derived-seed
answers are excluded. This cohort was used for tuning, so the test is exploratory.
"""

import argparse
import json
from math import comb
from pathlib import Path
from unittest.mock import patch

from compare_cloud_accuracy import PROMPT
from scoring import score_response
from steer import read_rows, summarize
from steer_jenny_six import main as run


CONDITIONS=[dict(name="baseline",interventions=[]),
            dict(name="L12-acoustic-a1",interventions=[dict(layer=12,group="acoustic",strength=1)]),
            dict(name="L16-answer-a1.25",interventions=[dict(layer=16,group="answer",strength=1.25)])]


def merge_cache(runs,records,vectors,conditions=None):
    selected={r["clip_id"]:r for r in records}
    expected={c["name"]:c["interventions"] for c in (conditions or CONDITIONS)}
    stat=vectors.stat()
    cached={}
    for folder in runs:
        if not (folder/"run_config.json").exists():
            print(f"No cache at {folder}; its missing answers will generate normally.",flush=True)
            continue
        config=json.loads((folder/"run_config.json").read_text())
        settings=config["settings"]
        if (Path(settings["vectors"]).resolve()!=vectors.resolve()
                or settings["vector_bytes"]!=stat.st_size or settings["vector_mtime_ns"]!=stat.st_mtime_ns
                or settings.get("seed_policy")!="constant_before_every_generation"
                or settings.get("seed")!=42 or settings.get("mode")!="fixed" or settings.get("style")!="angry"):
            print(f"Skipping incompatible vector/seed-policy cache: {folder}",flush=True)
            continue
        source={r["clip_id"]:r for r in config["recordings"]}
        targets={c["name"]:c.get("interventions",[]) for c in settings["conditions"]}
        for row in read_rows(folder/"results.jsonl"):
            cid,name=row["clip_id"],row["condition"]
            if cid not in selected or name not in expected:
                continue
            if any(source[cid][k]!=selected[cid][k] for k in ["question_id","audio_path","answer"]):
                raise ValueError(f"Cached recording differs: {folder}: {cid}")
            if (row.get("seed")!=42 or row.get("prompt")!=PROMPT or row.get("max_new_tokens")!=512
                    or row.get("decoder")!="fast_dllm" or row.get("mode")!="fixed"
                    or row.get("vector_style")!="angry" or targets.get(name)!=expected[name]
                    or row.get("interventions")!=expected[name]):
                raise ValueError(f"Cached condition or generation recipe differs: {folder}: {cid}/{name}")
            if str(row["reference_answer"])!=str(selected[cid]["answer"]):
                raise ValueError(f"Cached reference differs: {folder}: {cid}")
            checked={**row,**score_response(row["response"],selected[cid]["answer"])}
            key=(cid,name)
            if key in cached and cached[key]!=checked:
                raise ValueError(f"Conflicting repeated seed42 answer: {folder}: {key}")
            cached[key]=checked
    return cached


def report(output):
    config=json.loads((output/"run_config.json").read_text())
    conditions=config["settings"]["conditions"]
    rows={(r["clip_id"],r["condition"]):r for r in read_rows(output/"results.jsonl")}
    common=[r["clip_id"] for r in config["recordings"] if all((r["clip_id"],c["name"]) in rows for c in conditions)]
    print(f"Completed generations: {len(rows)}/{len(config['recordings'])*len(conditions)}")
    print(f"Same questions completed in every requested condition: {len(common)}/{len(config['recordings'])}")
    selected=[{**rows[cid,c["name"]],**score_response(rows[cid,c["name"]]["response"],rows[cid,c["name"]]["reference_answer"])}
              for cid in common for c in conditions]
    summary=summarize(selected,conditions)
    if common and any(c["name"]=="L12-acoustic-a1" for c in conditions):
        answers={(r["clip_id"],r["condition"]):r["correct"] for r in selected}
        w=sum(answers[cid,"L16-answer-a1.25"] and not answers[cid,"L12-acoustic-a1"] for cid in common)
        l=sum(answers[cid,"L12-acoustic-a1"] and not answers[cid,"L16-answer-a1.25"] for cid in common)
        d=w+l
        p=min(1.,2*sum(comb(d,k) for k in range(min(w,l)+1))/2**d)
        print(f"Direct L16 versus L12: {100*(w-l)/len(common):+.1f} pp; L16-only correct={w}, L12-only correct={l}; exact McNemar p={p:.4f}")
    print("Full cohort includes all original neutral/angry outcome groups. Layer selection used its subsets; this is exploratory.")
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run",type=Path,default=Path("outputs/steer-jenny-angry-complete"))
    p.add_argument("--vectors",type=Path,default=Path("outputs/jenny-angry-vectors/style-vectors.pt"))
    p.add_argument("--reuse-runs",type=Path,nargs="*",default=[
        Path("outputs/steer-jenny-recovery74-promoted-a1.25-seed42"),
        Path("outputs/steer-jenny-L16-L23-followup-seed42/regressions"),
        Path("outputs/steer-jenny-L12-vs-L16-full460-seed42")])
    p.add_argument("--l16-only",action="store_true",help="Complete only L16 plus its baseline; reuse the already tested subsets.")
    p.add_argument("--output",type=Path)
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--summary-only",action="store_true")
    args=p.parse_args()
    args.output=args.output or Path("outputs/steer-jenny-L16-full460-seed42" if args.l16_only else "outputs/steer-jenny-L12-vs-L16-full460-seed42")
    if args.summary_only:
        report(args.output)
        return
    conditions=[c for c in CONDITIONS if c["name"]!="L12-acoustic-a1"] if args.l16_only else CONDITIONS
    source=json.loads((args.source_run/"run_config.json").read_text())
    records=source["recordings"]
    if len(records)!=460 or len({r["question_id"] for r in records})!=460:
        p.error("Expected the original complete460 question cohort")
    stat=args.vectors.stat()
    old=source["settings"]
    if (args.vectors.resolve()!=Path(old["vectors"]).resolve()
            or stat.st_size!=old["vector_bytes"] or stat.st_mtime_ns!=old["vector_mtime_ns"]):
        p.error("Original vector checkpoint changed")
    if any(r["style"]!="neutral" or not Path(r["audio_path"]).is_absolute()
           or not Path(r["audio_path"]).is_file() for r in records):
        p.error("Expected neutral audio at its saved absolute paths")
    try:
        cached=merge_cache(args.reuse_runs,records,args.vectors,conditions)
    except ValueError as exc:
        p.error(str(exc))
    plan=dict(recordings=records,conditions=conditions,seed=42,
              selection="Full original460 cohort, without selection by answer correctness or classifier emotion",
              source_run=str(args.source_run.resolve()))
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/"selected-plan.json"
    if path.exists() and json.loads(path.read_text())!=plan:
        p.error("Frozen plan changed; use another --output")
    path.write_text(json.dumps(plan,indent=2)+"\n")
    command=["--previous",str(args.source_run),"--vectors",str(args.vectors),
             "--prepared-plan",str(path),"--output",str(args.output)]
    run(command+["--prepare-only"])
    results_path=args.output/"results.jsonl"
    saved={(r["clip_id"],r["condition"]):r for r in read_rows(results_path)}
    with results_path.open("a") as stream:
        copied=0
        for key,row in cached.items():
            if key in saved:
                if saved[key]!=row:
                    p.error(f"Existing answer conflicts with cached answer: {key}")
                continue
            stream.write(json.dumps(row)+"\n")
            stream.flush()
            copied+=1
    print(f"Imported {copied} compatible seed42 answers; prior derived-seed answers are not reused.",flush=True)
    if args.prepare_only:
        return
    with patch("steer_jenny_six.report",side_effect=lambda rows,cs,rs:summarize(rows,cs)):
        run(command)
    report(args.output)


if __name__=="__main__":
    main()
