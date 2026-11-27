"""Kimi spoken PBIF, paired neutral/angry Jenny readings of both constraint orders."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
import wave

import kimi_pbif as flipped
import pbif_emotions as pbif
from steer import read_rows
from winning_diffa_sweeps import atomic_text, digest, freeze

OUTPUT = Path("outputs/kimi-pbif-spoken-neutral-angry")


def source_cache():
    cache = flipped.OUTPUT / "upstream"
    return cache if cache.is_dir() else OUTPUT / "upstream"


def prepare(synthesize):
    from evaluate_kimi_audio import MODEL, REVISION, SOURCE_REVISION, TOKENIZER_MODEL, TOKENIZER_REVISION
    from kimi_bbh_layer_diagnostic import PARAMS
    from run_spoken_ifeval import PROMPT

    cache = source_cache()
    pbif.sources(cache)
    bank = pbif.question_bank(cache)
    pbif.registry(cache, prepare=True)
    questions = [{"question_id": f"{q}/{order}", "question": item["prompts"][order]}
                 for q, item in bank.items() for order in pbif.ORDERS]
    inputs = OUTPUT / "questions.jsonl"
    atomic_text(inputs, "".join(json.dumps(q) + "\n" for q in questions))
    audio = OUTPUT / "audio"
    command = [sys.executable, "-u", str(Path(__file__).with_name("cloud_tts_smoke.py")),
               "--provider", "azure", "--input-jsonl", str(inputs), "--num-examples", str(len(questions)),
               "--azure-voice", "en-US-JennyNeural", "--azure-intensity", "1.5", "--azure-rate-offset", "0",
               "--styles", *pbif.STYLES, "--output", str(audio), "--retry-passes", "5", "--resume"]
    subprocess.run(command if synthesize else [*command, "--prepare-only"], check=True)
    if not synthesize:
        return
    records = []
    for clip in read_rows(audio / "plan.jsonl"):
        q, order = clip["question_id"].split("/")
        path = audio / clip["audio_path"]
        if clip["question"] != bank[q]["prompts"][order]:
            raise ValueError(f"Spoken text differs for {q}/{order}")
        with wave.open(str(path)) as wav:
            frames, rate = wav.getnframes(), wav.getframerate()
            if (wav.getnchannels(), wav.getsampwidth(), rate) != (1, 2, 24000) or not frames:
                raise ValueError(f"Unexpected Azure PCM WAV: {path}")
            wav.setpos(frames - 1)
            if len(wav.readframes(1)) != 2:
                raise ValueError(f"Incomplete WAV: {path}")
        style = clip["style"]
        records.append(dict(question_id=q, condition=f"{order}/{style}", order=order, style=style,
                            voice=clip["voice"], spoken_prompt=clip["question"], audio_file=str(path.resolve()),
                            audio_sha256=digest(path), seconds=frames / rate))
    expected = {(r["question_id"], style) for r in questions for style in pbif.STYLES}
    keys = {(f"{r['question_id']}/{r['order']}", r["style"]) for r in records}
    if keys != expected or len(records) != len(expected):
        raise ValueError("Incomplete spoken PBIF recording pairs")
    plan = dict(protocol="kimi-pbif-spoken-v1", task="pbif", model=MODEL, revision=REVISION,
                source_revision=SOURCE_REVISION, tokenizer_model=TOKENIZER_MODEL, tokenizer_revision=TOKENIZER_REVISION,
                seed=42, params=dict(PARAMS, max_new_tokens=1024), data_revision=pbif.HF_REV,
                scorer_revision=pbif.CODE_REV, prompt=PROMPT, bank=bank, records=records)
    freeze(OUTPUT / "run_config.json", plan)
    lengths = [r["seconds"] for r in records]
    print(f"{len(records)} full recordings, {sum(s > 30 for s in lengths)} over 30 s, longest {max(lengths):.1f} s.")


def check_history(history, path, context_limit, budget):
    with wave.open(str(path)) as wav:
        # Both encoders use a 1,280-sample stride after resampling to 16 kHz.
        numerator = wav.getnframes() * 16000
        denominator = wav.getframerate() * 1280
        expected = (numerator + denominator - 1) // denominator
    audio_ids, _, mask, _, _ = history.to_tensor()
    features, = history.continuous_feature
    count = int(mask.sum())
    if count != expected or features.shape[1] != expected:
        raise ValueError(f"{path}: expected {expected} audio positions, got GLM {count}, Whisper {features.shape[1]}")
    if audio_ids.shape[1] + budget > context_limit:
        raise ValueError(f"{path}: input plus {budget} response tokens exceeds Kimi context {context_limit}")
    return audio_ids[mask.bool()], features


@contextmanager
def long_audio_guard(model, plan):
    from evaluate_kimi_audio import messages

    native = model.prompt_manager.get_prompt
    limit = model.alm.config.max_position_embeddings
    budget = plan["params"]["max_new_tokens"]

    def checked(chats, **kwargs):
        path, = [m["content"] for m in chats if m["message_type"] == "audio"]
        history = native(chats, **kwargs)
        check_history(history, path, limit, budget)
        return history

    longest = max(plan["records"], key=lambda r: r["seconds"])
    if longest["seconds"] > 30:
        path = longest["audio_file"]
        def encode(file):
            history = native(messages(file, plan["prompt"]), output_type="text")
            return check_history(history, file, limit, budget)

        with TemporaryDirectory() as tmp, wave.open(path) as src:
            muted = Path(tmp) / "muted-tail.wav"
            boundary = 30 * src.getframerate()
            with wave.open(str(muted), "wb") as dst:
                dst.setparams(src.getparams())
                dst.writeframes(src.readframes(boundary))
                dst.writeframes(bytes((src.getnframes() - boundary) * src.getsampwidth() * src.getnchannels()))
            ids, features = encode(path)
            muted_ids, muted_features = encode(muted)
            # 30 seconds at 12.5 audio positions per second.
            if ids[375:].equal(muted_ids[375:]) or features[:, 375:].equal(muted_features[:, 375:]):
                raise ValueError("Muting the longest recording after 30 s left an encoder's tail unchanged")
            print(f"Long-audio check passed: {longest['seconds']:.1f} s, {ids.numel()} audio positions; "
                  "both native encoders responded to audio after 30 s.", flush=True)
    with patch.object(model.prompt_manager, "get_prompt", checked):
        yield


def summary():
    from run_diffa_spoken_prompt import saved
    config = OUTPUT / "run_config.json"
    if not config.exists():
        audio = OUTPUT / "audio"
        print(f"Kimi spoken PBIF: {sum(1 for _ in audio.glob('*.wav'))}/792 recordings, awaiting generation.")
        return
    plan = json.loads(config.read_text())
    spoken = saved(OUTPUT, plan)
    verifier = pbif.registry(source_cache())
    pbif.report(OUTPUT, plan, spoken, verifier)
    base_config = flipped.OUTPUT / "run_config.json"
    if not spoken or not base_config.exists():
        return
    base = json.loads(base_config.read_text())
    for field in ("bank", "model", "revision", "source_revision", "params", "seed", "data_revision", "scorer_revision"):
        if base[field] != plan[field]:
            raise ValueError(f"Placement comparison has different {field}")
    written = saved(flipped.OUTPUT, base)
    spoken_scores = {(r["question_id"], r["order"], r["style"]): sum(r["checks"])
                     for r in read_rows(OUTPUT / "pbif-scores.jsonl")}
    contrasts = {}
    for order in pbif.ORDERS:
        differences = []
        for q, item in plan["bank"].items():
            keys = [(q, f"{order}/{style}") for style in pbif.STYLES]
            if not all(key in spoken and key in written for key in keys):
                continue
            neutral, angry = (sum(pbif.checks(item, verifier, written[key]["response"])) for key in keys)
            spoken_gap = spoken_scores[q, order, "angry"] - spoken_scores[q, order, "neutral"]
            differences.append((spoken_gap - (angry - neutral)) / 7)
        if differences:
            effect = pbif.paired_mean(differences)
            contrasts[order] = dict(n=len(differences), **effect)
            print(f"{order} spoken minus flipped change in angry-neutral constraint gap "
                  f"(N={len(differences)}): {effect['gain_pp']:+.2f} pp, 95% CI {effect['ci95_pp']}, p={effect['p']}")
    atomic_text(OUTPUT / "placement-summary.json", json.dumps(contrasts, indent=2) + "\n")


def main():
    from phi4_sweep import job_lock
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "synthesize", "generate", "summary"))
    mode = parser.parse_args().mode
    if mode == "summary":
        summary()
        return
    with job_lock(OUTPUT):
        if mode == "generate":
            from kimi_spoken_ifeval import generate
            generate(json.loads((OUTPUT / "run_config.json").read_text()), OUTPUT, input_guard=long_audio_guard)
        else:
            prepare(mode == "synthesize")


if __name__ == "__main__":
    main()
