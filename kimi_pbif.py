"""Replicate DIFFA's written PBIF requests and normal-rate Jenny carriers on Kimi."""
import argparse
import json
from hashlib import sha256
from pathlib import Path
import time
import zipfile

import pbif_emotions as pbif

STYLES = pbif.STYLES + pbif.ADDED_STYLES
OUTPUT = Path("outputs/kimi-pbif-five-styles")
BUNDLE = "kimi-pbif-inputs.zip"


def pack():
    """Export the exact five clips and checksummed scorer/data used on the DIFFA host."""
    from phi4_sweep import job_lock
    cache = pbif.PREVIOUS/"upstream"
    if not cache.is_dir():
        cache = pbif.OUTPUT/"upstream"
    pbif.sources(cache)
    bank = pbif.question_bank(cache)
    files = {}
    hashes = {}
    for folder, styles in ((pbif.OUTPUT, pbif.STYLES), (pbif.EXTENDED, pbif.ADDED_STYLES)):
        plan = json.loads((folder/"run_config.json").read_text())
        clips = pbif.normal_clips(folder, plan, None, styles=styles)
        if bank != plan["bank"] or pbif.question_records(bank, clips, styles) != plan["records"]:
            raise ValueError(f"Written prompts or audio conditions differ at {folder}")
        for (_, style), clip in clips.items():
            files[f"{style}.wav"] = Path(clip["audio_file"])
            hashes[style] = clip["audio_sha256"]
    bundle = Path("outputs")/BUNDLE
    with job_lock(bundle.parent/"kimi-pbif-export"):
        tmp = bundle.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, path in files.items():
                archive.write(path, name)
            for name in pbif.SOURCES:
                archive.write(cache/name, "upstream/"+name)
            archive.writestr("carriers.json", json.dumps(hashes))
        with zipfile.ZipFile(tmp) as archive:
            if archive.testzip() is not None:
                raise ValueError("PBIF bundle failed CRC verification")
        tmp.replace(bundle)
    print(f"Verified five original carrier WAVs and 198 paired written requests. Bundle: {bundle}")


def prepare(folder):
    from evaluate_kimi_audio import MODEL, REVISION, SOURCE_REVISION, TOKENIZER_MODEL, TOKENIZER_REVISION
    import kimi_bbh_layer_diagnostic as kimi
    cache = folder/"upstream"
    audio = folder/"audio"
    config = folder/"run_config.json"
    if config.exists():
        plan = json.loads(config.read_text())
        pbif.sources(cache)
        for name, checksum in {(r["audio_file"],r["audio_sha256"]) for r in plan["records"]}:
            if pbif.digest(name) != checksum:
                raise ValueError(f"Changed frozen carrier: {name}")
    else:
        bundle = next((p for p in (Path("/workspace")/BUNDLE, Path(BUNDLE), Path("outputs")/BUNDLE)
                       if p.is_file()), None)
        if bundle is None:
            raise FileNotFoundError(f"Place {BUNDLE} in /workspace (bash pbif.sh kimi pack on DIFFA).")
        cache.mkdir(parents=True, exist_ok=True)
        audio.mkdir(exist_ok=True)
        clips = {}
        with zipfile.ZipFile(bundle) as archive:
            hashes = json.loads(archive.read("carriers.json"))
            if set(hashes) != set(STYLES) or len(set(hashes.values())) != len(STYLES):
                raise ValueError("Expected five distinct DIFFA carrier recordings")
            for name, (_, checksum) in pbif.SOURCES.items():
                data = archive.read("upstream/"+name)
                if sha256(data).hexdigest() != checksum:
                    raise ValueError(f"PBIF release checksum mismatch: {name}")
                (cache/name).write_bytes(data)
            for style in STYLES:
                path = audio/f"{style}.wav"
                data = archive.read(path.name)
                if sha256(data).hexdigest() != hashes[style]:
                    raise ValueError(f"PBIF carrier checksum mismatch: {style}")
                path.write_bytes(data)
                clips[0,style] = dict(audio_file=str(path.resolve()), audio_sha256=hashes[style], voice="en-US-JennyNeural")
        bank = pbif.question_bank(cache)
        plan = dict(protocol="kimi-pbif-five-styles-rate0-v1", task="pbif", model=MODEL, revision=REVISION,
                    source_revision=SOURCE_REVISION, tokenizer_model=TOKENIZER_MODEL, tokenizer_revision=TOKENIZER_REVISION,
                    seed=42, params=dict(kimi.PARAMS, max_new_tokens=1024), data_revision=pbif.HF_REV,
                    scorer_revision=pbif.CODE_REV, bank=bank, records=pbif.question_records(bank, clips, STYLES))
    verifier = pbif.registry(cache, prepare=True)
    for item in plan["bank"].values():
        pbif.checks(item, verifier, "Preflight check.")
    if not config.exists():
        pbif.freeze(config, plan)
    print(f"Ready: {len(plan['bank'])} written requests, {len(plan['records'])} answers, five verified carriers.")


def wait_for_ifeval(folder):
    from run_diffa_spoken_prompt import saved
    config = folder/"run_config.json"
    print(f"Waiting for sad/terrified Kimi IFEval generation at {folder}. PBIF has not requested the GPU.", flush=True)
    last_count = None
    while True:
        if config.is_file():
            plan = json.loads(config.read_text())
            expected = {(q,s) for q in plan["bank"] for s in ("sad","terrified")}
            if len(expected) != 1016 or {(r["question_id"],r["condition"]) for r in plan["records"]} != expected:
                raise ValueError("Expected 508 matched sad/terrified Kimi IFEval pairs")
            rows = saved(folder, plan)
            if len(rows) != last_count:
                print(f"Kimi IFEval: {len(rows)}/1016 saved.", flush=True)
                last_count = len(rows)
            if rows.keys() == expected:
                return
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pack", "prepare", "run", "queued-run", "summary"))
    action = parser.parse_args().action
    if action == "pack":
        pack()
        return
    from phi4_sweep import job_lock
    from run_diffa_spoken_prompt import saved
    config = OUTPUT/"run_config.json"
    if action == "summary":
        if not config.is_file():
            print("Kimi PBIF: awaiting preparation.")
            return
        plan = json.loads(config.read_text())
        pbif.report(OUTPUT, plan, saved(OUTPUT, plan), pbif.registry(OUTPUT/"upstream"), STYLES)
        return
    with job_lock(OUTPUT):
        if action == "prepare":
            prepare(OUTPUT)
            return
        import kimi_spoken_ifeval as kimi
        if action == "queued-run":
            wait_for_ifeval(kimi.EXTRA_OUTPUT)
        kimi.generate(json.loads(config.read_text()), OUTPUT)


if __name__ == "__main__":
    main()
