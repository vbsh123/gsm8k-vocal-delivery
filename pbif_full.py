"""Extend the selected PBIF single-round split to every aligned row of both extreme orders."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import wave

import pbif_emotions as pbif
import kimi_pbif
import kimi_pbif_spoken
import diffa_pbif_spoken
import run_diffa_spoken_prompt as pilot
from run_reasoning_transfer import generate_audio, source_records
from run_spoken_ifeval import PROMPT
from winning_diffa_sweeps import atomic_text, digest, freeze

AUDIO = Path("outputs/pbif-full-spoken-audio")
PILOTS = {
    ("diffa", "flipped"): pbif.OUTPUT,
    ("diffa", "spoken"): diffa_pbif_spoken.OUTPUT,
    ("kimi", "flipped"): kimi_pbif.OUTPUT,
    ("kimi", "spoken"): kimi_pbif_spoken.OUTPUT,
}


def new_questions(bank, old):
    return {f"{q}/{order}": dict(question_id=f"{q}/{order}", question=item["prompts"][order], answer="")
            for q, item in bank.items() if q not in old for order in pbif.ORDERS}


def spoken_records(questions, audio):
    clips = source_records(audio, questions, pbif.STYLES)
    manifest = {}
    records = []
    for (key, style), clip in clips.items():
        q, order = key.split("/")
        with wave.open(clip["audio_file"]) as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 24000):
                raise ValueError(f"Unexpected Jenny WAV format: {key}/{style}")
            wav.setpos(wav.getnframes()-1)
            if len(wav.readframes(1)) != 2:
                raise ValueError(f"Truncated WAV: {key}/{style}")
        if clip["voice"] != "en-US-JennyNeural":
            raise ValueError(f"Unexpected voice: {key}/{style}")
        manifest[f"{key}/{style}"] = clip["audio_sha256"]
        records.append(dict(clip, question_id=q, condition=f"{order}/{style}", order=order,
                            prompt=PROMPT, spoken_prompt=questions[key]["question"]))
    return records, manifest


def scope(cache):
    bank = pbif.question_bank(cache, full=True)
    old = pbif.question_bank(cache)
    groups = {f"pbif_{i:04d}": req for req, rows in pbif.request_groups(cache).items() for i, _, _ in rows}
    if len(bank) != 2000 or len(old) != 198 or any(bank[q] != item for q, item in old.items()):
        raise ValueError("Full release and archived pilot differ")
    return bank, old, groups


def plan_extension(model, placement, base, bank, old):
    if (base["bank"] != old or base["data_revision"] != pbif.HF_REV or base["scorer_revision"] != pbif.CODE_REV
            or base["seed"] != 42):
        raise ValueError("Pilot protocol or cohort differs from the pinned PBIF release")
    if model == "kimi":
        from evaluate_kimi_audio import MODEL, REVISION, SOURCE_REVISION, TOKENIZER_REVISION, TOKENIZER_MODEL
        from kimi_bbh_layer_diagnostic import PARAMS
        if (base["model"], base["revision"], base["source_revision"], base["tokenizer_model"],
                base["tokenizer_revision"], base["params"]) != (MODEL, REVISION, SOURCE_REVISION, TOKENIZER_MODEL,
                                                               TOKENIZER_REVISION, dict(PARAMS, max_new_tokens=1024)):
            raise ValueError("Kimi model or decoding settings differ from the pilot")
    elif base["model"] != "diffa" or base["recipe"] != dict(pilot.evaluation.night.vanilla.RECIPE, max_new_tokens=1024):
        raise ValueError("DIFFA model or decoding settings differ from the pilot")
    extra = {q: item for q, item in bank.items() if q not in old}
    if placement == "flipped":
        records = pbif.question_records(extra, pbif.normal_clips(None, base, None), pbif.STYLES)
    else:
        if base.get("prompt", PROMPT) != PROMPT:
            raise ValueError("Changed spoken wrapper")
        records, manifest = spoken_records(new_questions(bank, old), AUDIO)
        if json.loads((AUDIO/"audio-sha256.json").read_text()) != manifest:
            raise ValueError("Spoken recordings differ from the shared audio manifest")
    plan = dict(base, protocol=f"{model}-{placement}-pbif-extension-v1", bank=extra, records=records,
                pilot_sha256=digest(PILOTS[model, placement]/"run_config.json"))
    if "audio_signature" in plan:
        plan["pilot_audio_signature"] = plan.pop("audio_signature")
    return plan


def describe(groups, scores, selected, order):
    cells = defaultdict(list)
    for q in selected:
        if all((q, order, s) in scores for s in pbif.STYLES):
            cells[groups[q]].append([scores[q, order, s] for s in pbif.STYLES])
    if not cells:
        return None
    neutral, angry = zip(*[[mean(values) for values in zip(*rows)] for rows in cells.values()])
    result = pbif.paired_mean([a-n for n, a in zip(neutral, angry)])
    return dict(result, base_requests=len(cells), rows=sum(map(len, cells.values())),
                neutral_pct=100*mean(neutral), angry_pct=100*mean(angry))


def summary(folder, base_rows, new_rows, bank, old, groups, verifier):
    scores = {}
    all_pass = {}
    lines = []
    for (q, condition), r in (base_rows | new_rows).items():
        if r["style"] not in pbif.STYLES:
            continue
        bits = pbif.checks(bank[q], verifier, r["response"])
        key = q, r["order"], r["style"]
        scores[key] = mean(bits)
        all_pass[key] = int(all(bits))
        lines.append(json.dumps(dict(question_id=q, condition=condition, checks=bits, response_tokens=r["response_tokens"])))
    print(f"PBIF neutral/angry: {len(scores)}/8000 answers, {len(new_rows)}/7208 extension answers.")
    results = {}
    for label, selected in (("full", bank), ("added", bank.keys()-old.keys()), ("pilot", old)):
        for order in pbif.ORDERS:
            for metric, values in (("constraints", scores), ("all_seven", all_pass)):
                stats = describe(groups, values, selected, order)
                if stats:
                    results[f"{label}/{order}/{metric}"] = stats
                    print(f"{label} {order} {metric}: {stats['rows']} matched rows, {stats['base_requests']} base requests; "
                          f"{stats['neutral_pct']:.2f}% -> {stats['angry_pct']:.2f}%, "
                          f"{stats['gain_pp']:+.2f} pp, 95% CI {stats['ci95_pp']}, p={stats['p']}")
    if len(scores) == 8000:
        atomic_text(folder/"pbif-scores.jsonl", "\n".join(lines)+"\n")
        atomic_text(folder/"summary.json", json.dumps(results, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("kimi", "diffa"))
    parser.add_argument("placement", choices=("spoken", "flipped"))
    parser.add_argument("mode", choices=("prepare", "audio", "build", "generate", "summary"))
    args = parser.parse_args()
    model, placement, mode = args.model, args.placement, args.mode
    original = PILOTS[model, placement]
    folder = Path(f"outputs/{model}-pbif-full-{placement}")
    config = folder/"run_config.json"
    if mode == "generate":
        with pilot.job_lock(folder):
            plan = json.loads(config.read_text())
            if plan["pilot_sha256"] != digest(original/"run_config.json"):
                raise ValueError("Pilot plan changed")
            if model == "kimi":
                from kimi_spoken_ifeval import generate
                generate(plan, folder, input_guard=kimi_pbif_spoken.long_audio_guard if placement == "spoken" else None)
            elif len(pilot.saved(folder, plan, repair=True)) < len(plan["records"]):
                from diffa_model_receipts import model_revisions
                revisions = json.loads((original/"model-revisions.json").read_text())
                if model_revisions() != revisions:
                    raise ValueError("DIFFA revisions differ from pilot")
                with pilot.gpu_queue():
                    from src.dataloader import build_chat_prompt
                    freeze(folder/"model-revisions.json", revisions)
                    pilot.generate(folder, plan, pilot.evaluation.night.DiffaBackend(), build_chat_prompt)
        return
    cache = next((d/"upstream" for d in (original, pbif.PREVIOUS, pbif.OUTPUT, kimi_pbif.OUTPUT)
                  if (d/"upstream"/"positive.jsonl").is_file()), folder/"upstream")
    pbif.sources(cache)
    bank, old, groups = scope(cache)
    if mode == "audio":
        if placement != "spoken":
            parser.error("Only spoken placement needs additional recordings")
        with pilot.job_lock(AUDIO):
            questions = new_questions(bank, old)
            generate_audio(AUDIO, list(questions.values()), pbif.STYLES)
            _, manifest = spoken_records(questions, AUDIO)
            freeze(AUDIO/"audio-sha256.json", manifest)
        return
    print("PBIF single_round/llama3_8b: all 2,000 source rows, 198 base requests, two orders and "
          "two styles. Pilot 198 rows; extension 1,802. Target 8,000 answers.")
    base_cfg = original/"run_config.json"
    if not base_cfg.is_file() and mode == "prepare":
        print(f"Pilot unavailable here: {original}. Use its machine for generation.")
        return
    base = json.loads(base_cfg.read_text())
    if base["bank"] != old:
        raise ValueError("Pilot cohort differs from release")
    cached = {key:r for key,r in pilot.saved(original, base).items() if r["style"] in pbif.STYLES}
    expected = {(q, f"{o}/{s}") for q in old for o in pbif.ORDERS for s in pbif.STYLES}
    if not cached.keys() <= expected:
        raise ValueError("Unexpected pilot conditions")
    print(f"Verified pilot answers available: {len(cached)}/792. Extension target: 7,208.")
    if mode == "prepare":
        return
    if mode == "build":
        pbif.registry(cache, prepare=True)
        if placement == "spoken" and not (AUDIO/"audio-sha256.json").is_file():
            raise SystemExit(f"Generate or transfer the shared {AUDIO} first (audio command).")
        with pilot.job_lock(folder):
            plan = plan_extension(model, placement, base, bank, old)
            freeze(config, plan)
        return
    if config.is_file():
        plan = json.loads(config.read_text())
        if plan["pilot_sha256"] != digest(base_cfg):
            raise ValueError("Pilot plan changed")
        added = pilot.saved(folder, plan)
    else:
        added = {}
    summary(folder, cached, added, bank, old, groups, pbif.registry(cache, prepare=True))


if __name__ == "__main__":
    main()
