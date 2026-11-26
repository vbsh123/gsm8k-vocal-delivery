"""Run native Kimi on the five existing Jenny written-IFEval carrier conditions."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile

import ifeval_carrier as carrier
import kimi_spoken_ifeval as spoken
from phi4_sweep import job_lock
from winning_diffa_sweeps import atomic_text, digest, freeze

STYLES = ("neutral", "angry", "happy", "sad", "terrified")
OUTPUT = Path("outputs/kimi-ifeval-written-carrier")
BUNDLE = "kimi-ifeval-flip-inputs.zip"


def bank():
    from steer import read_rows
    rows = read_rows(spoken.official.setup_official()/"data/input_data.jsonl")
    questions = {f"ifeval_{r['key']}":{k:r[k] for k in carrier.FIELDS} for r in rows}
    if len(questions) != 541:
        raise ValueError("Expected 541 official written requests.")
    return questions


def pack():
    questions = bank()
    files, expected = {}, {}
    parent = carrier.OUTPUT
    for suffix, styles in [("",carrier.STYLES), ("-happy-sad",carrier.ADDED_STYLES), ("-terrified",("terrified",))]:
        folder = parent.with_name(parent.name+suffix)
        config = folder/"run_config.json"
        plan = json.loads(config.read_text())
        if (plan["bank"] != questions or plan["revision"] != spoken.official.REVISION
                or plan["protocol"] != "diffa-ifeval-written-carrier-v1"):
            raise ValueError(f"Unexpected written flip source at {config}")
        clips = {}
        for r in plan["records"]:
            key = r["carrier_rate"], r["condition"]
            clip = {k:r[k] for k in ("voice","audio_file","audio_sha256")}
            if clips.setdefault(key,clip) != clip or clip["voice"] != "en-US-JennyNeural":
                raise ValueError(f"Unexpected Jenny carrier for {key}")
        if carrier.build_plan(questions, clips, styles)["records"] != plan["records"]:
            raise ValueError(f"Written prompts or carrier assignments differ at {config}")
        for (rate,style),clip in clips.items():
            name = f"rate{rate:+d}-{style}.wav"
            path = Path(clip["audio_file"])
            if digest(path) != clip["audio_sha256"]:
                raise ValueError(f"Changed source carrier at {path}")
            files[name] = path
            expected[name] = clip["audio_sha256"]
    if len(set(expected.values())) != 15:
        raise ValueError("Expected 15 distinct DIFFA recordings.")
    dest = Path("outputs")/BUNDLE
    with job_lock(parent.with_name("kimi-ifeval-flip-export")):
        tmp = dest.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("sha256.json", json.dumps(expected))
            for name,path in files.items():
                archive.write(path, name)
        tmp.replace(dest)
    print(f"Saved 15 verified carrier WAVs: {dest}")


def prepare():
    from evaluate_kimi_audio import MODEL, REVISION, SOURCE_REVISION, TOKENIZER_MODEL, TOKENIZER_REVISION
    config = OUTPUT/"run_config.json"
    frozen = json.loads(config.read_text()) if config.exists() else None
    audio = OUTPUT/"audio"
    audio.mkdir(parents=True, exist_ok=True)
    clips = {}
    if frozen:
        for r in frozen["records"]:
            clips[r["carrier_rate"],r["style"]] = {k:r[k] for k in ("audio_file","audio_sha256","voice")}
    else:
        bundle = next((p for p in (Path("/workspace")/BUNDLE, Path(BUNDLE), Path("outputs")/BUNDLE)
                       if p.is_file()), None)
        if bundle is None:
            raise FileNotFoundError(f"Place {BUNDLE} in /workspace. Export it on the DIFFA flip host with bash kimi_ifeval.sh flip pack.")
        with zipfile.ZipFile(bundle) as archive:
            expected = json.loads(archive.read("sha256.json"))
            for rate in carrier.RATES:
                for style in STYLES:
                    name = f"rate{rate:+d}-{style}.wav"
                    data = archive.read(name)
                    if sha256(data).hexdigest() != expected[name]:
                        raise ValueError(f"Carrier checksum mismatch: {name}")
                    path = audio/name
                    path.write_bytes(data)
                    clips[rate,style] = dict(audio_file=str(path.resolve()), audio_sha256=expected[name], voice="en-US-JennyNeural")
    if len({r["audio_sha256"] for r in clips.values()}) != 15:
        raise ValueError("Expected 15 distinct carrier recordings.")
    for clip in clips.values():
        if digest(clip["audio_file"]) != clip["audio_sha256"]:
            raise ValueError(f"Changed carrier: {clip['audio_file']}")
    questions = bank()
    records = carrier.build_plan(questions, clips, STYLES)["records"]
    plan = dict(protocol="kimi-ifeval-written-carrier-v1", task="ifeval", model=MODEL, revision=REVISION,
                source_revision=SOURCE_REVISION, tokenizer_model=TOKENIZER_MODEL, tokenizer_revision=TOKENIZER_REVISION,
                seed=42, params=dict(spoken.kimi.PARAMS, max_new_tokens=1024), official_revision=spoken.official.REVISION,
                primary_metric="strict_prompt", bank=questions, records=records)
    freeze(config, plan)
    print(f"Prepared 541 written IFEval requests, five styles, {len(records)} answers.")


def report(plan):
    evaluator = spoken.official.official_scorer()
    scores = spoken.report(plan, evaluator, OUTPUT, styles=STYLES, label="Kimi written IFEval")
    if not scores:
        return
    metrics = ("strict_prompt","loose_prompt")
    def scored(results):
        return dict(metrics={metric:results[metric.split("_")[0]]["prompt"] for metric in metrics})
    written = {key:scored(result) for key,result in scores.items()}
    ids = [q for q in plan["bank"] if (q,"neutral") in written]
    omnibus = {}
    for metric in metrics:
        omnibus[metric] = v = carrier.emotion_test(written, ids, STYLES[1:], metric)
        print(f"{metric} across angry/happy/sad/terrified: Q={v['q']:.2f}, asymptotic p={v['p_asymptotic']:.4g}")
    # Score both placements on the same prompts, using the same checkers.
    from run_diffa_spoken_prompt import saved
    heard = {}
    for folder in (spoken.OUTPUT,spoken.EXTRA_OUTPUT):
        config = folder/"run_config.json"
        if not config.is_file():
            continue
        old = json.loads(config.read_text())
        fields = ("model","revision","source_revision","tokenizer_model","tokenizer_revision","params","official_revision","seed")
        if any(old[k] != plan[k] for k in fields) or old["prompt"] != spoken.official.PROMPT:
            raise ValueError(f"Incompatible spoken IFEval reference at {folder}")
        for (q,s),row in saved(folder,old).items():
            if (q,s) in written:
                if any(old["bank"][q][k] != plan["bank"][q][k] for k in carrier.FIELDS):
                    raise ValueError(f"Scoring inputs differ for {q}")
                heard[q,s] = scored(spoken.official.score(plan["bank"][q],row["response"],evaluator))
    changes = {}
    common = [q for q in ids if all((q,s) in heard for s in STYLES)]
    if common:
        for s in STYLES[1:]:
            changes[s] = {}
            groups = {key:{q:data[q,style] for q in common} for key,data,style in (
                ("neutral",heard,"neutral"),("angry",heard,s),
                ("neutral-reasoning",written,"neutral"),("angry-reasoning",written,s))}
            for metric in metrics:
                v = carrier.accuracy_interaction(groups,metric)
                changes[s][metric] = v
                print(f"Written minus spoken change in {s}-neutral gap / {metric}, N={len(common)}: "
                      f"{v['interaction_pp']:+.2f} pp, CI={v['ci95_pp']}, p={v['p']}")
    atomic_text(OUTPUT/"placement-summary.json",json.dumps(dict(written_ids=ids,omnibus=omnibus,
                matched_spoken_ids=common,placement_changes=changes),indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("pack","prepare","generate","summary"))
    mode = p.parse_args().mode
    if mode == "pack":
        pack()
        return
    config = OUTPUT/"run_config.json"
    if mode == "summary":
        if config.exists():
            spoken.official.setup_official()
            report(json.loads(config.read_text()))
        else:
            print("Kimi written IFEval is awaiting preparation.")
        return
    with job_lock(OUTPUT):
        if mode == "prepare":
            prepare()
        else:
            spoken.generate(json.loads(config.read_text()), OUTPUT)


if __name__ == "__main__":
    main()
