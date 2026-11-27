"""Evaluate DIFFA on the same full spoken PBIF recordings used for Kimi."""
import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import wave
import zipfile

import pbif_emotions as pbif
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


def main():
    import ifeval_carrier as flip
    from diffa_model_receipts import model_revisions
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "summary"))
    mode = parser.parse_args().mode
    config = OUTPUT / "run_config.json"
    cache = next((p / "upstream" for p in (pbif.PREVIOUS, pbif.OUTPUT, OUTPUT)
                  if (p / "upstream").is_dir()), OUTPUT / "upstream")
    if mode == "summary" and not config.exists():
        print("DIFFA spoken PBIF is awaiting preparation.")
        return
    pbif.sources(cache)
    verifier = pbif.registry(cache, prepare=mode != "summary")
    if mode != "summary":
        with flip.pilot.job_lock(OUTPUT):
            bank = pbif.question_bank(cache)
            records = recordings(bank)
            recipe = dict(flip.night.vanilla.RECIPE, max_new_tokens=1024)
            plan = dict(protocol="diffa-spoken-pbif-v1", task="pbif", name="spoken PBIF", model="diffa",
                        seed=42, data_revision=pbif.HF_REV, scorer_revision=pbif.CODE_REV, recipe=recipe,
                        max_new_tokens=1024, bank=bank, records=records, audio_signature=AUDIO_SIGNATURE)
            pbif.freeze(config, plan)
            print(f"{len(records)} verified recordings; {sum(r['seconds'] > 30 for r in records)} over 30 s. "
                  f"Written wrapper: {PROMPT}", flush=True)
            if mode == "prepare":
                return
            if len(flip.pilot.saved(OUTPUT, plan, repair=True)) < len(records):
                with flip.pilot.gpu_queue():
                    from src.dataloader import build_chat_prompt
                    pbif.freeze(OUTPUT / "model-revisions.json", model_revisions())
                    flip.pilot.generate(OUTPUT, plan, flip.night.DiffaBackend(), build_chat_prompt)
    plan = json.loads(config.read_text())
    spoken = flip.pilot.saved(OUTPUT, plan)
    pbif.report(OUTPUT, plan, spoken, verifier)
    base_config = pbif.OUTPUT / "run_config.json"
    if spoken and base_config.is_file():
        base = json.loads(base_config.read_text())
        for field in ("model", "bank", "recipe", "data_revision", "scorer_revision"):
            if base[field] != plan[field]:
                raise ValueError(f"Flipped run has different {field}")
        pbif.report_placement(OUTPUT, plan, spoken, flip.pilot.saved(pbif.OUTPUT, base), verifier)


if __name__ == "__main__":
    main()
