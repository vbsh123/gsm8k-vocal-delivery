"""Diagnostic ×1 answer steering on disjoint Kimi spoken IFEval recovery prompts."""
import argparse
import csv
import io
import json
from pathlib import Path
import random
import time

import kimi_ifeval_collect as fit
from phi4_queued_workers import gpu_queue
from phi4_sweep import job_lock
from run_diffa_spoken_prompt import saved
from winning_diffa_sweeps import atomic_text, digest, freeze

OUT = Path("outputs/kimi-spoken-ifeval-recovery-layers")
spoken = fit.spoken
kimi = fit.kimi


def select(test_ids, scores):
    eligible = [q for q in test_ids if scores[q,"angry"] and not scores[q,"neutral"]]
    random.Random(42).shuffle(eligible)
    return eligible[:50], len(eligible)


def prepare():
    src = spoken.OUTPUT/"run_config.json"
    base = json.loads(src.read_text())
    with job_lock(fit.OUTPUT):
        fitting = fit.prepare()
    rows = saved(spoken.OUTPUT, base)
    spoken.official.setup_official()
    scorer = spoken.official.official_scorer()
    scores = {}
    for q in fitting["test_ids"]:
        for s in fit.STYLES:
            scores[q,s] = spoken.official.score(base["bank"][q],rows[q,s]["response"],scorer)["strict"]["prompt"]
    ids, count = select(fitting["test_ids"], scores)
    records = [r for r in base["records"] if r["question_id"] in ids and r["condition"] in fit.STYLES]
    if set(ids).intersection(fitting["collection_ids"]):
        raise ValueError("Recovery prompts overlap fitting IDs")
    plan = dict(protocol="kimi-ifeval-strict-recovery-layers-v1", source_sha256=digest(src),
                fit_config_sha256=digest(fit.OUTPUT/"run_config.json"), eligible=count, ids=ids,
                layers=fitting["layers"], strength=1., prompt=base["prompt"], params=base["params"],
                bank={q:base["bank"][q] for q in ids}, records=records,
                reference={q:{s:rows[q,s]["response"] for s in fit.STYLES} for q in ids})
    freeze(OUT/"run_config.json",plan)
    print(f"{len(ids)} selected from {count} strict angry-pass/neutral-fail prompts on the held-out half.", flush=True)
    return plan


def wait_jobs():
    from kimi_ifeval_flip import OUTPUT as flipped
    from kimi_pbif_spoken import EXTRA as pbif
    for folder, expected in ((flipped,2705),(pbif,1188)):
        path = folder/"deliveries.jsonl"
        previous = None
        while True:
            size = path.stat().st_size if path.is_file() else -1
            if size != previous:
                with path.open("rb") if size >= 0 else io.BytesIO() as f:
                    n = sum(line.endswith(b"\n") for line in f)
                print(f"Queue: {folder.name} has {n}/{expected} saved answers.",flush=True)
                if n >= expected:
                    base = json.loads((folder/"run_config.json").read_text())
                    if len(saved(folder,base)) != expected:
                        raise ValueError(f"Unexpected number of responses in {folder}")
                    break
                previous = size
            time.sleep(30)


def shifts(neutral, angry, steered, fit_vector):
    import torch
    observed = angry-neutral
    moved = steered-neutral
    def cos(a,b):
        denom = a.norm(dim=1)*b.norm(dim=1)
        return (a*b).sum(1)/denom.masked_fill(denom==0, float("nan"))
    return dict(shift_norm=moved.norm(dim=1), natural_gap_norm=observed.norm(dim=1),
                remaining_to_angry=(steered-angry).norm(dim=1), cosine_to_natural=cos(moved,observed),
                cosine_to_fit=cos(moved,fit_vector), natural_cosine_to_fit=cos(observed,fit_vector))


def generate(plan):
    import torch
    cfg = OUT/"run_config.json"
    receipt = digest(cfg)
    vp = fit.OUTPUT/"style-vectors.pt"
    v = torch.load(vp,map_location="cpu",weights_only=True)
    fit_plan = json.loads((fit.OUTPUT/"run_config.json").read_text())
    direction = v["mean_differences"]["angry"]
    if (digest(fit.OUTPUT/"run_config.json") != plan["fit_config_sha256"] or
            v["config_sha256"] != plan["fit_config_sha256"] or v["collection_ids"] != fit_plan["collection_ids"]
            or set(v["collection_ids"]).intersection(plan["ids"]) or v["layers"] != plan["layers"]
            or direction.shape != (28,3584) or not direction.isfinite().all()):
        raise ValueError("Incompatible fitting vector")
    freeze(OUT/"vector.json",dict(sha256=digest(vp), fit_config_sha256=v["config_sha256"]))
    natural = OUT/"natural"
    outputs = OUT/"steered"
    for d in (natural, outputs):
        d.mkdir(exist_ok=True)
    todo = [(q,l) for q in plan["ids"] for l in plan["layers"] if not (outputs/f"{q}-L{l}.pt").exists()]
    if not todo:
        return
    records = {(r["question_id"],r["condition"]):r for r in plan["records"]}
    with gpu_queue(),torch.inference_mode():
        model = kimi.load_model(Path("external/Kimi-Audio"))
        layers = kimi.decoder_layers(model)
        for q in plan["ids"]:
            pending = [l for i,l in todo if i==q]
            if not pending:
                continue
            traces = {}
            for s in fit.STYLES:
                r = records[q,s]
                if digest(r["audio_file"]) != r["audio_sha256"]:
                    raise ValueError(f"Changed input {q}/{s}")
                path = natural/f"{q}-{s}.pt"
                if not path.is_file():
                    trace = fit.capture(model,layers,r,plan)
                    kimi.atomic_save(path,dict(trace, config_sha256=receipt, question_id=q, style=s))
                t = torch.load(path,map_location="cpu",weights_only=True)
                if (t["config_sha256"],t["question_id"],t["style"]) != (receipt,q,s):
                    raise ValueError("Incompatible baseline replay")
                traces[s] = t["answer_means"]
            for layer in pending:
                # Capture installed inside the steering context observes the actual post-intervention output.
                with kimi.steer_layer(layers[layer],direction[layer]):
                    t = fit.capture(model,layers,records[q,"neutral"],plan)
                metrics = shifts(traces["neutral"],traces["angry"],t.pop("answer_means"),direction)
                kimi.atomic_save(outputs/f"{q}-L{layer}.pt",dict(t,metrics=metrics,layer=layer,
                                    question_id=q,config_sha256=receipt))
                print(f"{q} L{layer:02} ×1: {t['forwards']} forwards",flush=True)


def summary(plan):
    import torch
    receipt = digest(OUT/"run_config.json")
    print(f"{len(plan['ids'])} selected / {plan['eligible']} eligible, 28 separate layer trials.")
    if not (OUT/"steered").is_dir():
        print("Waiting for collection or GPU.")
        return
    spoken.official.setup_official()
    scorer = spoken.official.official_scorer()
    replays = {}
    replay_passes = {s:[] for s in fit.STYLES}
    for q in plan["ids"]:
        for s in fit.STYLES:
            path = OUT/"natural"/f"{q}-{s}.pt"
            if path.is_file():
                t = torch.load(path,map_location="cpu",weights_only=True)
                if (t["config_sha256"], t["question_id"], t["style"]) != (receipt,q,s):
                    raise ValueError(f"Stale natural replay: {path}")
                replays[q,s] = t["response"] == plan["reference"][q][s]
                replay_passes[s].append(spoken.official.score(plan["bank"][q],t["response"],scorer)["strict"]["prompt"])
    print(f"Natural replay text matches archived responses: {sum(replays.values())}/{len(replays)}.")
    for s, bits in replay_passes.items():
        print(f"Natural {s} replay: strict {sum(bits)}/{len(bits)}")
    table = io.StringIO()
    writer = csv.writer(table)
    keys = ("shift_norm","natural_gap_norm","remaining_to_angry","cosine_to_natural","cosine_to_fit","natural_cosine_to_fit")
    writer.writerow(["intervention_layer","measured_layer","n","strict_pass","loose_pass",*keys])
    for layer in plan["layers"]:
        metrics = []
        strict = loose = 0
        for q in plan["ids"]:
            path = OUT/"steered"/f"{q}-L{layer}.pt"
            if not path.is_file():
                continue
            t = torch.load(path,map_location="cpu",weights_only=True)
            if (t["config_sha256"],t["question_id"],t["layer"]) != (receipt,q,layer):
                raise ValueError(f"Stale generation: {path}")
            scores = spoken.official.score(plan["bank"][q],t["response"],scorer)
            strict += scores["strict"]["prompt"]
            loose += scores["loose"]["prompt"]
            metrics.append(t["metrics"])
        if metrics:
            avg = {k:torch.stack([m[k] for m in metrics]).nanmean(0) for k in keys}
            for readout in plan["layers"]:
                writer.writerow([layer,readout,len(metrics),strict,loose,*[avg[k][readout].item() for k in keys]])
            print(f"L{layer:02} ×1: strict recovered {strict}/{len(metrics)}, loose passed {loose}/{len(metrics)}")
    atomic_text(OUT/"layer-comparison.csv",table.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","wait","generate","summary"))
    action = parser.parse_args().action
    if action=="wait":
        wait_jobs()
    elif action=="summary":
        cfg = OUT/"run_config.json"
        if cfg.exists():
            summary(json.loads(cfg.read_text()))
        else:
            print("Awaiting preparation.")
    else:
        with job_lock(OUT):
            if action=="prepare":
                prepare()
            else:
                generate(json.loads((OUT/"run_config.json").read_text()))


if __name__=="__main__":
    main()
