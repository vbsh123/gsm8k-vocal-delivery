"""Evaluate DIFFA on the same full spoken PBIF recordings used for Kimi."""
import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import wave
import zipfile

import pbif_emotions as pbif
import kimi_pbif_spoken as kimi
from run_spoken_ifeval import PROMPT
from steer import read_rows

OUTPUT = Path("outputs/diffa-pbif-spoken-neutral-angry")
AUDIO = Path("outputs/kimi-pbif-spoken-neutral-angry/audio")
BUNDLE = "kimi-pbif-spoken-audio.zip"
AUDIO_SIGNATURE = "16c8cfcb68ab0a09dcd9b81abee4dd225d524e6e51622fe7c3091ae6f6a5c50e"


def recordings(bank):
    questions = [(q, order) for q in bank for order in pbif.ORDERS]
    names = [f"q{i:05d}-azure-{s}.wav" for i in range(len(questions)) for s in pbif.STYLES]
    files = ["plan.jsonl", *names]
    missing = [name for name in files if not (AUDIO / name).is_file()]
    if missing:
        archive = next((p for p in (Path("/workspace") / BUNDLE, Path(BUNDLE)) if p.is_file()), None)
        if archive is None:
            raise FileNotFoundError(f"Upload {BUNDLE} to /workspace or supply the full {AUDIO} directory.")
        AUDIO.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for name in missing:
                path = AUDIO / name
                tmp = path.with_suffix(".part")
                tmp.write_bytes(z.read(str(AUDIO / name)))
                tmp.replace(path)

    clips = read_rows(AUDIO / "plan.jsonl")
    if len(clips) != len(names):
        raise ValueError("Unexpected number of clips in the spoken manifest")
    records, signatures = [], []
    for i, ((q, order), style) in enumerate(product(questions, pbif.STYLES)):
        clip = clips[i]
        text = bank[q]["prompts"][order]
        if (clip["question_id"], clip["style"], clip["audio_path"], clip["question"], clip["provider"], clip["voice"]) != (
                f"{q}/{order}", style, names[i], text, "azure", "en-US-JennyNeural"):
            raise ValueError(f"Spoken PBIF manifest mismatch for {q}/{order}/{style}")
        path = AUDIO / names[i]
        checksum = pbif.digest(path)
        signatures.append([clip["question_id"], style, checksum])
        with wave.open(str(path)) as wav:
            seconds = wav.getnframes() / wav.getframerate()
        records.append(dict(question_id=q, order=order, style=style, condition=f"{order}/{style}",
                            voice=clip["voice"], prompt=PROMPT, spoken_prompt=text, seconds=seconds,
                            audio_file=str(path.resolve()), audio_sha256=checksum))
    if sha256(json.dumps(sorted(signatures)).encode()).hexdigest() != AUDIO_SIGNATURE:
        raise ValueError("Audio differs from the verified Kimi spoken PBIF bundle")
    return records


EXTRA = Path("outputs/diffa-pbif-spoken-added-emotions")
EXTRA_BUNDLE = "pbif-spoken-happy-sad-terrified.zip"


def pack_emotions():
    """Export the exact inputs once Kimi has frozen its audio plan, even during inference."""
    from tempfile import NamedTemporaryFile
    config = kimi.EXTRA / "run_config.json"
    if not config.is_file():
        count = sum(1 for _ in (kimi.EXTRA/"audio").glob("*.wav"))
        raise SystemExit(f"{count}/1188 WAVs present. Pack after synthesis finishes and writes {config}.")
    plan = json.loads(config.read_text())
    expected = set(product(plan["bank"], pbif.ORDERS, kimi.ADDED))
    if len(expected) != 1188 or plan["protocol"] != "kimi-pbif-spoken-v1" or plan["prompt"] != PROMPT:
        raise ValueError("Unexpected Kimi spoken PBIF plan.")
    dest = Path("outputs")/EXTRA_BUNDLE
    with NamedTemporaryFile(dir=dest.parent, suffix=".zip.tmp", delete=False) as temp:
        tmp = Path(temp.name)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            manifest = []
            for record in plan["records"]:
                key = record["question_id"], record["order"], record["style"]
                expected.remove(key)
                question, order, style = key
                if (record["condition"] != f"{order}/{style}" or record["voice"] != "en-US-JennyNeural"
                        or record["spoken_prompt"] != plan["bank"][question]["prompts"][order]):
                    raise ValueError(f"Unexpected source record for {key}")
                data = Path(record["audio_file"]).read_bytes()
                if sha256(data).hexdigest() != record["audio_sha256"]:
                    raise ValueError(f"Changed Kimi audio for {key}")
                name = f"{question}_{order}_{style}.wav"
                archive.writestr(name, data)
                manifest.append(dict(record, audio_file=name))
            if expected:
                raise ValueError(f"Missing {len(expected)} Kimi recordings")
            archive.writestr("records.json", json.dumps(manifest))
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"Saved {len(manifest)} Kimi-verified recordings to {dest}", flush=True)


def added_recordings(bank, folder):
    bundle = next((root/EXTRA_BUNDLE for root in (Path("/workspace"), Path.cwd(), Path("outputs"))
                   if (root/EXTRA_BUNDLE).is_file()), None)
    if bundle is None:
        raise FileNotFoundError(f"Place {EXTRA_BUNDLE} in /workspace after Kimi audio synthesis.")
    expected = set(product(bank, pbif.ORDERS, kimi.ADDED))
    audio = folder/"audio"
    audio.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(bundle) as archive:
        for source in json.loads(archive.read("records.json")):
            q, order, style = source["question_id"], source["order"], source["style"]
            expected.remove((q, order, style))
            name = f"{q}_{order}_{style}.wav"
            if (source["audio_file"] != name or source["condition"] != f"{order}/{style}"
                    or source["voice"] != "en-US-JennyNeural" or source["spoken_prompt"] != bank[q]["prompts"][order]):
                raise ValueError(f"Unmatched PBIF recording for {q}/{order}/{style}")
            path = audio/name
            if not path.is_file() or pbif.digest(path) != source["audio_sha256"]:
                data = archive.read(name)
                if sha256(data).hexdigest() != source["audio_sha256"]:
                    raise ValueError(f"Source WAV checksum mismatch: {name}")
                tmp = path.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(path)
            records.append(dict(source, audio_file=str(path.resolve()), prompt=PROMPT))
    if expected:
        raise ValueError(f"Missing {len(expected)} additional PBIF recordings")
    return records


def main():
    import ifeval_carrier as flip
    from diffa_model_receipts import model_revisions
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "summary"))
    parser.add_argument("--emotions", action="store_true")
    args = parser.parse_args()
    mode = args.mode
    extra = args.emotions
    folder = EXTRA if extra else OUTPUT
    config = folder / "run_config.json"
    cache = next((p / "upstream" for p in (pbif.PREVIOUS, pbif.OUTPUT, OUTPUT)
                  if (p / "upstream").is_dir()), OUTPUT / "upstream")
    if mode == "summary" and not config.exists():
        print(f"{folder.name} is awaiting preparation.")
        return
    pbif.sources(cache)
    verifier = pbif.registry(cache, prepare=mode != "summary")
    if mode != "summary":
        with flip.pilot.job_lock(folder):
            bank = pbif.question_bank(cache)
            if extra and config.is_file():
                records = json.loads(config.read_text())["records"]
                for record in records:
                    if pbif.digest(record["audio_file"]) != record["audio_sha256"]:
                        raise ValueError(f"Changed DIFFA input at {record['audio_file']}")
            else:
                records = added_recordings(bank, folder) if extra else recordings(bank)
            recipe = dict(flip.night.vanilla.RECIPE, max_new_tokens=1024)
            plan = dict(protocol="diffa-spoken-pbif-v1", task="pbif", name="spoken PBIF", model="diffa",
                        seed=42, data_revision=pbif.HF_REV, scorer_revision=pbif.CODE_REV, recipe=recipe,
                        max_new_tokens=1024, bank=bank, records=records, audio_signature=AUDIO_SIGNATURE)
            if extra:
                baseline_cfg = OUTPUT/"run_config.json"
                base = json.loads(baseline_cfg.read_text())
                for field in ("model", "recipe", "seed", "data_revision", "scorer_revision", "bank"):
                    if base[field] != plan[field]:
                        raise ValueError(f"Spoken baseline differs in {field}")
                plan.update(protocol="diffa-spoken-pbif-added-v1", baseline_sha256=pbif.digest(baseline_cfg),
                            audio_signature=sha256(json.dumps(sorted((r["condition"], r["question_id"], r["audio_sha256"])
                                                                     for r in records)).encode()).hexdigest())
            pbif.freeze(config, plan)
            print(f"{len(records)} verified recordings; {sum(r['seconds'] > 30 for r in records)} over 30 s. "
                  f"Written wrapper: {PROMPT}", flush=True)
            if mode == "prepare":
                return
            if len(flip.pilot.saved(folder, plan, repair=True)) < len(records):
                with flip.pilot.gpu_queue():
                    from src.dataloader import build_chat_prompt
                    revisions = model_revisions()
                    if extra and revisions != json.loads((OUTPUT/"model-revisions.json").read_text()):
                        raise ValueError("DIFFA model revision differs from the spoken baseline")
                    pbif.freeze(folder / "model-revisions.json", revisions)
                    flip.pilot.generate(folder, plan, flip.night.DiffaBackend(), build_chat_prompt)
    plan = json.loads(config.read_text())
    spoken = flip.pilot.saved(folder, plan)
    styles = pbif.STYLES
    if extra:
        base_cfg = OUTPUT/"run_config.json"
        if pbif.digest(base_cfg) != plan["baseline_sha256"]:
            raise ValueError("Spoken baseline plan changed")
        base = json.loads(base_cfg.read_text())
        print(f"New happy/sad/terrified responses: {len(spoken)}/1188")
        spoken.update(flip.pilot.saved(OUTPUT, base))
        plan = dict(plan, records=base["records"] + plan["records"])
        styles += kimi.ADDED
    pbif.report(folder, plan, spoken, verifier, styles)
    if extra and spoken:
        kimi.emotion_test(folder, plan["bank"], styles[1:])
    base_config = pbif.OUTPUT / "run_config.json"
    if spoken and base_config.is_file():
        base = json.loads(base_config.read_text())
        for field in ("model", "bank", "recipe", "data_revision", "scorer_revision"):
            if base[field] != plan[field]:
                raise ValueError(f"Flipped run has different {field}")
        written = flip.pilot.saved(pbif.OUTPUT, base)
        extended = pbif.EXTENDED/"run_config.json"
        if extra and extended.is_file():
            additions = json.loads(extended.read_text())
            if additions["baseline_sha256"] != pbif.digest(base_config):
                raise ValueError("Flipped additional emotions reference a different baseline")
            written.update(flip.pilot.saved(pbif.EXTENDED, additions))
        pbif.report_placement(folder, plan, spoken, written, verifier, styles)


if __name__ == "__main__":
    main()
