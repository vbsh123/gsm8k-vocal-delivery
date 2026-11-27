"""Run natural terrified DIFFA on the archived 508 spoken IFEval prompt IDs."""
import argparse
import json
from pathlib import Path

import diffa_ifeval_x1 as original
import run_spoken_ifeval as official
from ifeval_carrier import emotion_test
from rescore_audio_steering import paired
from run_diffa_spoken_prompt import saved
from steer import read_rows
from winning_diffa_sweeps import atomic_text, digest, freeze

OUT = Path("outputs/diffa-spoken-ifeval508-terrified")
REFERENCE = Path("data/ifeval_spoken508_natural_baseline.json")
STYLES = ("angry", "happy", "sad", "terrified")


def prepare():
    fit, test, ref = original.inputs()
    ids = fit + test
    baseline = json.loads(REFERENCE.read_text())
    if set(baseline) != set(ids):
        raise ValueError("Paired baseline and recorded prompt IDs differ.")
    hashes = json.loads(REFERENCE.with_name("ifeval_spoken508_terrified_sha256.json").read_text())
    if set(hashes) != set(ids):
        raise ValueError("Terrified audio manifest differs from the 508 prompt split.")
    for q in ids:
        if ref["audio"][q].setdefault("terrified", hashes[q]) != hashes[q]:
            raise ValueError(f"Terrified audio hash differs from original fitting reference at {q}.")
    audio = original.recordings("terrified", ids, ref)
    examples = read_rows(official.setup_official() / "data/input_data.jsonl")
    bank = {f"ifeval_{e['key']}": e for e in examples}
    plan = dict(protocol="diffa-natural-terrified-spoken-ifeval508-v1", model="diffa",
                prompt=official.PROMPT, max_new_tokens=1024, seed=42, revision=official.REVISION,
                baseline_sha256=digest(REFERENCE), bank={q:bank[q] for q in ids},
                records=[dict(audio[q], style="terrified", condition="terrified") for q in ids])
    freeze(OUT / "run_config.json", plan)
    return plan


def summary(plan):
    if digest(REFERENCE) != plan["baseline_sha256"]:
        raise ValueError("The four-style reference has changed.")
    baseline = json.loads(REFERENCE.read_text())
    rows = saved(OUT, plan)
    print(f"DIFFA natural spoken IFEval: {len(rows)}/508 terrified responses.")
    if not rows:
        return
    evaluator = official.official_scorer()
    ids = [q for q in plan["bank"] if (q, "terrified") in rows]
    bits = {}
    scored = []
    for q in ids:
        record = rows[q, "terrified"]
        result = official.score(plan["bank"][q], record["response"], evaluator)
        scored.append(dict(record, official_scores=result))
        for style, flags in baseline[q].items():
            bits[q,style] = dict(metrics=flags)
        bits[q,"terrified"] = dict(metrics={f"{m}_prompt": result[m]["prompt"] for m in ("strict","loose")})
    comparisons, omnibus = {}, {}
    for metric in ("strict_prompt", "loose_prompt"):
        pairs = [(bits[q,"neutral"]["metrics"][metric], bits[q,"terrified"]["metrics"][metric]) for q in ids]
        stat = comparisons[metric] = paired(pairs)
        qtest = omnibus[metric] = emotion_test(bits, ids, STYLES, metric)
        print(f"{metric} neutral {stat['baseline_correct']}/{len(ids)} -> terrified {stat['steered_correct']}/{len(ids)}, "
              f"{stat['gain_pp']:+.2f} pp, 95% CI {stat['ci95_gain_pp']}, exact p={stat['mcnemar_exact_p']:.4g}")
        print(f"Across {'/'.join(STYLES)}: Q={qtest['q']:.2f}, df={qtest['df']}, asymptotic p={qtest['p_asymptotic']:.4g}")
    atomic_text(OUT/"official-scored.jsonl","".join(json.dumps(r)+"\n" for r in scored))
    atomic_text(OUT/"summary.json",json.dumps(dict(matched=ids,comparisons=comparisons,omnibus=omnibus),indent=2)+"\n")


def run(plan):
    from diffa_happy_nightly import generate
    done = saved(OUT, plan, repair=True)
    todo = [r for r in plan["records"] if (r["question_id"],"terrified") not in done]
    if not todo:
        return
    evaluator = official.official_scorer()
    for record in todo:
        official.score(plan["bank"][record["question_id"]], "Scorer check.", evaluator)
    checksum = digest(OUT/"run_config.json")
    with original.gpu_queue():
        from extract_vectors import load_model
        model, tokenizer, features = load_model()
        with (OUT/"deliveries.jsonl").open("a") as stream:
            for i,record in enumerate(todo,len(done)+1):
                response, encoding = generate(record, plan, plan["prompt"], model, tokenizer, features)
                tokens = len(tokenizer.encode(response, add_special_tokens=False))
                row = dict(record, model="diffa", prompt=plan["prompt"], response=response,
                           response_tokens=tokens, audio_encoding=encoding, run_config_sha256=checksum)
                stream.write(json.dumps(row)+"\n")
                stream.flush()
                print(f"{i}/508 {record['question_id']} terrified {tokens} tokens", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["run","prepare","summary"])
    action = parser.parse_args().action
    if action == "summary":
        config = OUT/"run_config.json"
        if config.exists():
            summary(json.loads(config.read_text()))
        else:
            print("DIFFA natural terrified is awaiting preparation.")
    else:
        with original.job_lock(OUT):
            plan = prepare()
            print("508 verified terrified recordings; natural DIFFA spoken IFEval.", flush=True)
            if action == "run":
                run(plan)
                summary(plan)
