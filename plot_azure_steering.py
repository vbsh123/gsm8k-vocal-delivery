"""Build a standalone interactive Azure steering graph (standard library only).

Without --runs, uses the Azure neutral-input tables pasted in this conversation:
the 50-question L0–25 sweep, 460-pair follow-up, and partial reverse test on 373.
With --runs, reads actual saved configurations/results and intersects completed
questions across plotted strengths/groups at each layer. Older TTS is rejected.
The chart retains only cohorts with more than 50 paired questions.
"""

import argparse
from collections import defaultdict
from html import escape
import json
from math import comb
from pathlib import Path

from scoring import score_response


def point(group, strength, correct, n, wins, losses):
    d = wins+losses
    p = min(1.0, 2*sum(comb(d,k) for k in range(min(wins,losses)+1))/2**d) if d else 1.0
    return dict(group=group, strength=strength, correct=correct, n=n, wins=wins, losses=losses, p=p)


def diagnostic_data():
    recovery = [(5,"acoustic",1,13),(3,"answer",2,15),(11,"answer",1,24),
                (0,"answer",1,17),(2,"acoustic",1,12),(3,"answer",1,16),
                (8,"answer",1,12),(9,"answer",1,13),(12,"acoustic",1,22),
                (14,"answer",1,15),(23,"acoustic",1,4),(11,"answer",1.5,19)]
    regressions = [(11,"answer",1,15),(12,"acoustic",1,9),
                   (11,"answer",1.5,20),(0,"answer",1,13)]
    layers = {}
    for layer,group,strength,recovered in recovery:
        layers.setdefault(str(layer),dict(n=74,baseline=0,points=[]))["points"].append(
            point(group,strength,recovered,74,recovered,0))
    other = {}
    for layer,group,strength,lost in regressions:
        other.setdefault(str(layer),dict(n=260,baseline=260,points=[]))["points"].append(
            point(group,strength,260-lost,260,0,lost))
    return [dict(id="recovery-74",title="Recovery diagnostic · 74 baseline failures",layers=layers,
                 source="Outcome-selected: angry succeeded and neutral failed; these 74 also failed the fresh baseline. Accuracy here is recovery, not overall model accuracy. P-values are conditional diagnostics."),
            dict(id="regressions-260",title="Regression diagnostic · 260 baseline successes",layers=other,
                 source="Outcome-selected: neutral and angry both succeeded, and all 260 succeeded under the fresh baseline. Accuracy here measures retention of successes, not overall model accuracy. Do not combine directly with the recovery cohort.")]


def pasted_data():
    # Full Azure pilot table supplied by the user: same 50 recordings at every layer.
    acoustic = [(1,3),(0,1),(1,1),(0,1),(0,1),(2,1),(0,1),(0,2),(0,1),(1,3),
                (0,4),(0,2),(1,2),(0,3),(0,2),(0,2),(1,2),(0,1),(0,2),(0,4),
                (1,2),(0,1),(0,1),(1,1),(1,2),(0,1)]
    answer = [(1,3),(0,2),(0,0),(1,0),(1,2),(0,1),(0,0),(0,2),(1,3),(1,2),
              (0,3),(1,3),(0,3),(0,1),(1,2),(0,2),(0,3),(0,4),(0,4),(0,2),
              (0,1),(0,2),(0,2),(0,3),(0,1),(0,3)]
    sweep = {str(layer):dict(n=50,baseline=37,points=[
        point(group,1,37+w-l,50,w,l)
        for group,(w,l) in [("acoustic",acoustic[layer]),("answer",answer[layer])]])
        for layer in range(26)}
    return [dict(title="Batch 1 vector → batch 2 neutrals · L0–25 pilot", layers=sweep,
                 source="Pasted full Azure sweep: the same 50 neutral recordings, baseline 37/50, acoustic and answer steering at +1 on layers 0–25. Distinct from the larger follow-ups."),
            dict(id="batch1-neutral-487",title="Batch 2 vector → batch 1 neutrals · 487 pairs", source="Latest pasted Azure table; 487 completed neutral pairs. Replaces the earlier 373-pair snapshot. Other deliveries excluded.",
                 layers={"11":dict(n=487, baseline=319, points=[point("answer",1,330,487,30,19)]),
                         "12":dict(n=487, baseline=319, points=[point("acoustic",1,327,487,25,17)])}),
            dict(id="batch2-neutral-460",title="Batch 1 vector → batch 2 neutrals · complete", source="Pasted Azure table; all 460 matched neutral–angry questions. Steering is applied to neutral audio.",
                 layers={"11":dict(n=460, baseline=304, points=[point("answer",1,313,460,30,21)]),
                         "12":dict(n=460, baseline=304, points=[point("acoustic",1,316,460,29,17)])})]+diagnostic_data()


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def from_runs(paths):
    experiments = {}
    for path in paths:
        config = json.loads((path/"run_config.json").read_text())
        settings = config["settings"]
        records = {r["clip_id"]:r for r in config["recordings"] if r.get("style")=="neutral"}
        if not records or any("jenny" not in str(r.get("speaker", "")).lower() for r in records.values()):
            raise ValueError(f"Expected Azure Jenny neutral recordings: {path}")
        if "jenny" not in settings["vectors"].lower():
            raise ValueError(f"Expected an Azure Jenny vector checkpoint: {path}")
        key = json.dumps([settings["vectors"],settings["vector_bytes"],settings["vector_mtime_ns"],
                          settings["seed"],settings["mode"],sorted(records)],sort_keys=True)
        exp = experiments.setdefault(key, dict(settings=settings, records=records, rows={}, conditions={}, sources=[]))
        exp["sources"].append(path.name)
        for c in settings["conditions"]:
            targets = c.get("interventions")
            if targets is None:
                targets = [] if c["name"]=="baseline" else [{k:c[k] for k in ("layer","group","strength")}]
            # A coefficient curve is defined for combinations with the same
            # coefficient at every intervention. Mixed strengths need a separate plot.
            if not targets or len({t["strength"] for t in targets})==1:
                exp["conditions"][c["name"]] = targets
        latest = {(r["clip_id"],r["condition"]):r for r in read_rows(path/"results.jsonl")}
        for k, r in latest.items():
            if k[0] not in records or k[1] not in exp["conditions"]:
                continue
            ok = score_response(r["response"],r["reference_answer"])["correct"]
            value = dict(ok=ok, seed=r["seed"], prompt=r["prompt"], decoder=r["decoder"], tokens=r["max_new_tokens"])
            if k in exp["rows"] and exp["rows"][k] != value:
                raise ValueError(f"Conflicting cached answer/recipe: {path}: {k}")
            exp["rows"][k] = value
    datasets = []
    for exp in experiments.values():
        rows, conditions = exp["rows"], exp["conditions"]
        by_layer = defaultdict(list)
        for name, targets in conditions.items():
            if targets and -2 <= targets[0]["strength"] <= 2:
                layer=" + ".join(str(v) for v in sorted({t["layer"] for t in targets}))
                for group in sorted({t["group"] for t in targets}):
                    by_layer[layer].append((name,dict(layer=layer,group=group,strength=targets[0]["strength"],label=name)))
        layers = {}
        for layer, targets in sorted(by_layer.items()):
            common = {cid for cid,c in rows if c=="baseline"}
            available = [(name,t) for name,t in targets if any(c==name for _,c in rows)]
            for name,_ in available:
                common &= {cid for cid,c in rows if c==name}
            if not common or not available:
                continue
            b = {cid:rows[cid,"baseline"] for cid in common}
            points = []
            for name,t in available:
                for cid in common:
                    r = rows[cid,name]
                    if any(r[k]!=b[cid][k] for k in ("seed","prompt","decoder","tokens")):
                        raise ValueError(f"Nonmatching baseline recipe at {cid}: {name}")
                wins=sum(rows[cid,name]["ok"] and not b[cid]["ok"] for cid in common)
                losses=sum(b[cid]["ok"] and not rows[cid,name]["ok"] for cid in common)
                points.append({**point(t["group"],t["strength"],sum(rows[cid,name]["ok"] for cid in common),len(common),wins,losses),"label":name})
            layers[str(layer)] = dict(n=len(common),baseline=sum(r["ok"] for r in b.values()),points=points)
        if layers:
            vector=Path(exp["settings"]["vectors"]).parent.name
            datasets.append(dict(title=f"{vector} · {len(exp['records'])} planned neutrals", layers=layers,
                                 source="Saved Azure runs: "+", ".join(exp["sources"])+". Same completed questions across plotted coefficients at each layer."))
    if not datasets:
        raise ValueError("No completed single-layer Azure steering points")
    return datasets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,nargs="+")
    p.add_argument("--output",type=Path,default=Path("outputs/azure-steering-coefficients.html"))
    p.add_argument("--split-output",type=Path,help="Write a separate standalone HTML per cohort into this directory.")
    args=p.parse_args()
    data=from_runs(args.runs) if args.runs else pasted_data()
    data=[{**e,"layers":{k:v for k,v in e["layers"].items() if v["n"]>50}}
          for e in data]
    data=[e for e in data if e["layers"]]
    if not data:
        p.error("No Azure cohorts with more than 50 paired questions")
    template=Path(__file__).with_name("azure_steering_graph.html").read_text()
    if args.split_output:
        args.split_output.mkdir(parents=True,exist_ok=True)
        for index,e in enumerate(data,1):
            payload=json.dumps([e],ensure_ascii=True).replace("<","\\u003c")
            html=template.replace("__AZURE_STEERING_DATA__",payload)
            html=html.replace('<div class="field"><label for="experiment">',
                              '<div class="field" style="display:none"><label for="experiment">')
            html=html.replace("Anger steering: strength vs accuracy",escape(e["title"]))
            html=html.replace("Azure anger steering — accuracy by coefficient",escape(e["title"]))
            path=args.split_output/(e.get("id",f"azure-experiment-{index}")+".html")
            path.write_text(html)
            print(f"Saved {path.resolve()}")
        return
    payload=json.dumps(data,ensure_ascii=True).replace("<","\\u003c")
    html=template.replace("__AZURE_STEERING_DATA__",payload)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(html)
    print(f"Saved {args.output.resolve()} ({len(data)} separate Azure experiments)")


if __name__=="__main__":
    main()
