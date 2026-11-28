"""Transfer DIFFA's spoken COLLIE pilot to Kimi, or run the paired written condition on DIFFA."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile, ZIP_DEFLATED

import collie_spoken as collie
import pbif_emotions as pbif
import run_diffa_spoken_prompt as pilot
from steer import read_rows
from winning_diffa_sweeps import digest, freeze

ARCHIVE = "collie-spoken190-inputs.zip"
OUTPUT = {"kimi": Path("outputs/kimi-collie-spoken"), "diffa": Path("outputs/diffa-collie-flipped")}


def validate_spoken(plan, selected):
    if plan["protocol"] != "diffa-collie-spoken-v1" or plan["revision"] != collie.REV:
        raise ValueError("Wrong source protocol or COLLIE revision")
    expected = [(q, style, family, item["prompt"]) for q, family, item in selected for style in collie.STYLES]
    actual = [(r["question_id"], r["style"], r["family"], r["spoken_prompt"]) for r in plan["records"]]
    if actual != expected or any(r["condition"] != r["style"] or r["prompt"] != collie.PROMPT
                                 or r["voice"] != "en-US-JennyNeural" for r in plan["records"]):
        raise ValueError("Source WAV manifest differs from the 190-prompt paired spoken protocol")


def pack():
    source = collie.OUT
    config = source/"run_config.json"
    plan = json.loads(config.read_text())
    validate_spoken(plan, collie.select(collie.release(source/"upstream")))
    dest = Path("outputs")/ARCHIVE
    with NamedTemporaryFile(dir=dest.parent, delete=False) as temp:
        tmp = Path(temp.name)
    try:
        with ZipFile(tmp, "w", ZIP_DEFLATED, compresslevel=1) as z:
            z.write(config, "spoken-plan.json")
            for name in collie.FILES:
                z.write(source/"upstream"/name, name)
            for r in plan["records"]:
                data = Path(r["audio_file"]).read_bytes()
                if sha256(data).hexdigest() != r["audio_sha256"]:
                    raise ValueError(f"Changed source recording for {r['question_id']} / {r['style']}")
                z.writestr(f"audio/{r['question_id']}-{r['style']}.wav", data)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"380 verified WAVs and pinned COLLIE inputs: {dest.resolve()}")


def copy_member(z, name, path, checksum):
    if path.exists() and digest(path) == checksum:
        return
    data = z.read(name)
    if sha256(data).hexdigest() != checksum:
        raise ValueError(f"Invalid archive member: {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def transfer(folder):
    from evaluate_kimi_audio import MODEL, REVISION, SOURCE_REVISION, TOKENIZER_MODEL, TOKENIZER_REVISION
    from kimi_bbh_layer_diagnostic import PARAMS
    bundle = next((p/ARCHIVE for p in (Path("/workspace"), Path("outputs"), Path(".")) if (p/ARCHIVE).is_file()), None)
    if bundle is None:
        raise FileNotFoundError(f"Upload {ARCHIVE} to /workspace on the Kimi machine")
    with ZipFile(bundle) as z:
        for name, checksum in collie.FILES.items():
            copy_member(z, name, folder/"upstream"/name, checksum)
        source = z.read("spoken-plan.json")
        base = json.loads(source)
        validate_spoken(base, collie.select(collie.release(folder/"upstream")))
        records = []
        for r in base["records"]:
            name = f"audio/{r['question_id']}-{r['style']}.wav"
            path = folder/name
            copy_member(z, name, path, r["audio_sha256"])
            records.append(dict(r, audio_file=str(path.resolve())))
    return dict(protocol="kimi-spoken-collie-v1", name="Kimi spoken COLLIE", model=MODEL, revision=REVISION,
                source_revision=SOURCE_REVISION, tokenizer_model=TOKENIZER_MODEL, tokenizer_revision=TOKENIZER_REVISION,
                collie_revision=collie.REV, spoken_plan_sha256=sha256(source).hexdigest(), seed=42,
                prompt=collie.PROMPT, params=dict(PARAMS, max_new_tokens=1024), records=records)


def flipped(folder):
    from pbif_placement_vectors import CARRIER_HASHES, OUT as vector_dir
    from kimi_pbif import BUNDLE
    source = collie.OUT
    path = source/"run_config.json"
    base = json.loads(path.read_text())
    selected = collie.select(collie.release(source/"upstream"))
    validate_spoken(base, selected)
    clips = {}
    for style, checksum in CARRIER_HASHES.items():
        wav = vector_dir/f"{style}.wav"
        if not wav.is_file():
            bundle = next((root/BUNDLE for root in (Path("/workspace"), Path("outputs"), Path(".")) if (root/BUNDLE).is_file()), None)
            if bundle is None:
                raise FileNotFoundError(f"Expected {wav} or {BUNDLE} on the spoken COLLIE machine")
            wav = folder/"audio"/f"{style}.wav"
            with ZipFile(bundle) as z:
                copy_member(z, f"{style}.wav", wav, checksum)
        if digest(wav) != checksum:
            raise ValueError(f"Changed original PBIF {style} carrier WAV")
        clips[style] = dict(audio_file=str(wav.resolve()), audio_sha256=checksum)
    records = [dict(question_id=qid, family=family, style=style, condition=style, prompt=item["prompt"],
                    spoken_prompt=pbif.PHRASE, voice="en-US-JennyNeural", carrier_rate=0, **clips[style])
               for qid, family, item in selected for style in collie.STYLES]
    recipe = dict(pilot.evaluation.night.vanilla.RECIPE, max_new_tokens=1024)
    if base["recipe"] != recipe:
        raise ValueError("DIFFA decoding settings differ from the spoken run")
    return dict(protocol="diffa-flipped-collie-v1", name="DIFFA flipped COLLIE", model="diffa", task="collie",
                revision=collie.REV, spoken_plan_sha256=digest(path), seed=42, max_new_tokens=1024, recipe=recipe, records=records)


def report(model, folder, plan):
    cache = (collie.OUT if model == "diffa" else folder)/"upstream"
    selected = collie.select(collie.release(cache))
    scored = collie.report(plan, selected, pilot.saved(folder, plan), folder)
    baseline = collie.OUT/"scores.jsonl"
    if model == "diffa" and scored and baseline.is_file():
        if digest(collie.OUT/"run_config.json") != plan["spoken_plan_sha256"]:
            raise ValueError("Source spoken run changed")
        spoken = {r["question_id"]: r for r in read_rows(baseline)}
        if set(spoken) != {q for q, _, _ in selected}:
            raise ValueError("Incomplete source scores")
        shifts = [(r["angry"]-r["neutral"]) - (spoken[r["question_id"]]["angry"]-spoken[r["question_id"]]["neutral"])
                  for r in scored]
        result = pbif.paired_mean(shifts)
        print(f"Written minus spoken change in angry–neutral gap on {len(shifts)} paired problems: "
              f"{result['gain_pp']:+.2f} pp, 95% CI {result['ci95_pp']}, p={result['p']}")


def snapshot(model):
    folder = OUTPUT[model]
    folders = [collie.OUT, folder] if model == "diffa" else [folder]
    selected = collie.select(collie.release(folders[0]/"upstream"))
    from collie.constraints import All

    destination = folder.parent/f"collie-{model}-results.zip"
    with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        tmp = Path(temporary.name)
    try:
        with ZipFile(tmp, "w", ZIP_DEFLATED) as archive:
            for run in folders:
                config = run/"run_config.json"
                rows = pilot.saved(run, json.loads(config.read_text()))
                scores = []
                paired_counts = dict.fromkeys(collie.STYLES, 0)
                paired_total = 0
                for qid, family, item in selected:
                    checker, targets = item["constraint"], item["targets"]
                    paired = all((qid, style) in rows for style in collie.STYLES)
                    for style in collie.STYLES:
                        if (qid, style) not in rows:
                            continue
                        response = rows[qid, style]["response"]
                        passed = bool(checker.check(response, targets))
                        flags = ([bool(f(response, t)) for f, t in zip(checker.callables, targets, strict=True)]
                                 if isinstance(checker, All) else [passed])
                        if all(flags) != passed:
                            raise ValueError(f"Component scores disagree with released scorer for {qid}")
                        scores.append(dict(question_id=qid, family=family, condition=style, prompt_pass=passed,
                                           component_pass=flags, targets=targets))
                        if paired:
                            paired_counts[style] += sum(flags)
                            if style == collie.STYLES[0]:
                                paired_total += len(flags)
                archive.write(config, f"{run.name}/run_config.json")
                for filename, items in [("deliveries.jsonl", rows.values()), ("component-scores.jsonl", scores)]:
                    archive.writestr(f"{run.name}/{filename}", "".join(json.dumps(r)+"\n" for r in items))
                print(f"{run.name}: {len(rows)} saved answers. Matched component passes: " +
                      ", ".join(f"{s} {n}/{paired_total}" for s, n in paired_counts.items()))
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)
    print(destination.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("diffa", "kimi"))
    parser.add_argument("mode", choices=("pack", "results", "prepare", "generate", "summary"))
    args = parser.parse_args()
    if args.mode == "results":
        return snapshot(args.model)
    if args.mode == "pack":
        return pack()
    folder = OUTPUT[args.model]
    config = folder/"run_config.json"
    if args.mode == "summary":
        if config.is_file():
            report(args.model, folder, json.loads(config.read_text()))
        else:
            print(f"{folder.name}: awaiting preparation (190/2,080 release items; 380 answers planned).")
        return
    with pilot.job_lock(folder):
        if args.mode == "prepare":
            if config.is_file():
                for r in json.loads(config.read_text())["records"]:
                    if digest(r["audio_file"]) != r["audio_sha256"]:
                        raise ValueError(f"Changed frozen recording: {r['audio_file']}")
            else:
                freeze(config, transfer(folder) if args.model == "kimi" else flipped(folder))
            print(f"{folder.name}: 190/2,080 release items, 13 families, 380 neutral/angry answers planned.")
            return
        plan = json.loads(config.read_text())
        if args.model == "kimi":
            from kimi_pbif_spoken import long_audio_guard
            from kimi_spoken_ifeval import generate
            generate(plan, folder, input_guard=long_audio_guard)
        elif len(pilot.saved(folder, plan, repair=True)) < len(plan["records"]):
            from diffa_model_receipts import model_revisions
            revisions = model_revisions()
            if revisions != json.loads((collie.OUT/"model-revisions.json").read_text()):
                raise ValueError("Spoken and flipped DIFFA model revisions differ")
            with pilot.gpu_queue():
                from src.dataloader import build_chat_prompt
                freeze(folder/"model-revisions.json", revisions)
                pilot.generate(folder, plan, pilot.evaluation.night.DiffaBackend(), build_chat_prompt)


if __name__ == "__main__":
    main()
