"""PBIF single-round written requests with neutral and angry Jenny audio at normal speed."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import mean, median, stdev
import sys
from urllib.request import urlopen

from rescore_audio_steering import paired
from winning_diffa_sweeps import atomic_text, digest, freeze

PREVIOUS = Path("outputs/diffa-pbif-emotions-v2")
OUTPUT = Path("outputs/diffa-pbif-neutral-angry-rate0")
PHRASE = "Complete the task provided in the text."
PROTOCOL = "diffa-pbif-neutral-angry-rate0-v1"
STYLES = ("neutral", "angry")
ORDERS = ("tau+1", "tau-1")
HF_REV = "da7734bb9a940e89d30d8dd93e62209710f9893b"
CODE_REV = "dd4149e5a1bb6b084d870f403066d02ff18bf2b0"
HF = f"https://huggingface.co/datasets/donadona/PBIF/resolve/{HF_REV}/single_round/llama3_8b/"
CODE = f"https://raw.githubusercontent.com/meowpass/PBIF/{CODE_REV}/code/evaluation/"
SOURCES = {
    "positive.jsonl": (HF+"llama3_8b_tau1.0.jsonl", "53cf1d6b1dedc58f89ae3930481e4c1ebe12c8f43bdda7f3b8013e67f986218a"),
    "negative.jsonl": (HF+"llama3_8b_tau-1.0.jsonl", "54d5962f5b73ff82ec32bd42b63b632baacfd5bbe82f2ca7709eb61b7c6e47df"),
    "instructions.py": (CODE+"instructions.py", "26a925e9c2e7b3791b4b2a8508be687201c280dc0d8bc90e0fa798b56c836a4e"),
    "instructions_registry.py": (CODE+"instructions_registry.py", "1510429c4832811059de2cfb24ef20991d8252cdd55fcb54c7f11ab4683bd83c"),
    "instructions_util.py": (CODE+"instructions_util.py", "82bb3339329bd4c5e9f173ceb3889619129c9f7e8ec7261d97c8bbf90b2f31c4"),
}


def sources(cache):
    cache.mkdir(parents=True, exist_ok=True)
    for name, (url, checksum) in SOURCES.items():
        path = cache/name
        if not path.exists():
            with urlopen(url, timeout=90) as stream:
                data = stream.read()
            if sha256(data).hexdigest() != checksum:
                raise ValueError(f"Download checksum mismatch for {url}")
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
        elif digest(path) != checksum:
            raise ValueError(f"Changed PBIF source: {path}")


def question_bank(cache):
    by_request = defaultdict(list)
    with (cache/"positive.jsonl").open() as pos, (cache/"negative.jsonl").open() as neg:
        for index, lines in enumerate(zip(pos, neg, strict=True)):
            a, b = map(json.loads, lines)
            fields = ("instruction_id_list", "kwargs", "constraints")
            if any(a[key] != b[key] or len(a[key]) != 7 for key in fields):
                raise ValueError(f"Unmatched constraints at row {index}")
            prefixes = []
            for row, ranking in ((a, list(range(7))), (b, list(reversed(range(7))))):
                suffix = ". ".join(row["constraints"][i] for i in ranking)+"."
                if row["ranking"] != ranking or not row["prompt"].endswith(suffix):
                    raise ValueError(f"Unexpected written order at row {index}")
                prefixes.append(row["prompt"].removesuffix(suffix))
            if prefixes[0] != prefixes[1]:
                raise ValueError(f"Unmatched request at row {index}")
            by_request[prefixes[0].strip()].append((index, a, b))
    if len(by_request) != 198 or sum(map(len, by_request.values())) != 2000:
        raise ValueError("Unexpected PBIF release size")
    rng = random.Random(42)
    cohort = [rng.choice(by_request[q]) for q in sorted(by_request)]
    rng.shuffle(cohort)
    return {f"pbif_{index:04d}": dict(prompts=dict(zip(ORDERS, [a["prompt"], b["prompt"]])),
                instruction_id_list=a["instruction_id_list"], kwargs=a["kwargs"], constraints=a["constraints"])
            for index, a, b in cohort}


def normal_clips(folder, previous, flip):
    if previous is None:
        return flip.carrier_clips(folder/"audio", STYLES, phrase=PHRASE, rates=(0,))
    clips = {}
    for row in previous["records"]:
        style = row["style"]
        if row["carrier_rate"] != 0 or style not in STYLES:
            continue
        clip = {k: row[k] for k in ("voice", "audio_file", "audio_sha256")}
        if (clip["voice"] != "en-US-JennyNeural" or row["spoken_prompt"] != PHRASE
                or clips.setdefault((0, style), clip) != clip):
            raise ValueError(f"Inconsistent normal-speed {style} carrier in the previous plan")
    if set(clips) != {(0, s) for s in STYLES} or len({c["audio_sha256"] for c in clips.values()}) != 2:
        raise ValueError("Expected distinct normal-speed neutral and angry clips")
    for clip in clips.values():
        if digest(clip["audio_file"]) != clip["audio_sha256"]:
            raise ValueError(f"Changed carrier: {clip['audio_file']}")
    return clips


def reuse_answers(folder, plan, previous, pilot):
    if previous is None or not (PREVIOUS/"deliveries.jsonl").exists():
        return
    for k in ("model", "recipe", "data_revision", "scorer_revision", "max_new_tokens", "bank"):
        if plan[k] != previous[k]:
            raise ValueError(f"Cannot reuse PBIF answers with different {k}")
    earlier = pilot.saved(PREVIOUS, previous)
    done = pilot.saved(folder, plan, repair=True)
    checksum = digest(folder/"run_config.json")
    copied = []
    for record in plan["records"]:
        key = record["question_id"], record["condition"]
        row = earlier.get(key)
        if key not in done and row and all(row[k] == value for k, value in record.items()):
            copied.append(dict(row, source_config_sha256=row["run_config_sha256"], run_config_sha256=checksum))
    if copied:
        freeze(folder/"model-revisions.json", json.loads((PREVIOUS/"model-revisions.json").read_text()))
        with (folder/"deliveries.jsonl").open("a") as stream:
            for row in copied:
                stream.write(json.dumps(row)+"\n")
        print(f"Reused {len(copied)} matching normal-speed answers.")


def make_plan(bank, clips, flip):
    records = []
    for q, item in bank.items():
        for order in ORDERS:
            for style in STYLES:
                clip = clips[0, style]
                records.append(dict(question_id=q, condition=f"{order}/{style}", order=order, style=style,
                                    prompt=item["prompts"][order], carrier_rate=0, spoken_prompt=PHRASE,
                                    **{k: clip[k] for k in ("voice", "audio_file", "audio_sha256")}))
    return dict(protocol=PROTOCOL, task="pbif", name="pbif", model="diffa", seed=42,
                bank=bank, records=records, max_new_tokens=1024, recipe=dict(flip.night.vanilla.RECIPE, max_new_tokens=1024),
                data_revision=HF_REV, scorer_revision=CODE_REV)


def registry(cache, prepare=False):
    import nltk
    from langdetect import DetectorFactory
    try:
        nltk.word_tokenize("Test.")
    except LookupError:
        if not prepare:
            raise
        for resource in ("punkt_tab", "punkt"):
            nltk.download(resource, raise_on_error=True)
        nltk.word_tokenize("Test.")
    DetectorFactory.seed = 0
    sys.path.insert(0, str(cache.resolve()))
    import instructions_registry
    for name in ("instructions", "instructions_registry", "instructions_util"):
        if Path(sys.modules[name].__file__).resolve() != (cache/(name+".py")).resolve():
            raise ValueError(f"Expected the pinned PBIF implementation of {name}")
    return instructions_registry.INSTRUCTION_DICT


def checks(item, verifier, response):
    bits = []
    for key, kwargs in zip(item["instruction_id_list"], item["kwargs"], strict=True):
        instruction = verifier[key](key)
        instruction.build_description(**kwargs)
        bits.append(bool(response.strip() and instruction.check_following(response)))
    return bits


def paired_mean(differences):
    """Question-level paired mean difference and two-sided t interval/test."""
    from scipy.stats import t
    n = len(differences)
    estimate = 100*mean(differences)
    if n < 2 or len(set(differences)) == 1:
        return dict(gain_pp=estimate, ci95_pp=None, p=None)
    se = 100*stdev(differences)/n**0.5
    margin = float(t.ppf(.975, n-1))*se
    return dict(gain_pp=estimate, ci95_pp=[estimate-margin, estimate+margin],
                p=float(2*t.sf(abs(estimate)/se, n-1)))


def report(folder, plan, saved, verifier):
    ids = [q for q in plan["bank"] if all((q, f"{o}/{s}") in saved for o in ORDERS for s in STYLES)]
    print(f"PBIF: {len(saved)}/{len(plan['records'])} answers, {len(ids)}/{len(plan['bank'])} four-way matched requests.")
    if not ids:
        return
    results, bits, fractions, scored = {}, {}, {}, []
    for o in ORDERS:
        for s in STYLES:
            rows = [saved[q, f"{o}/{s}"] for q in ids]
            flags = [checks(plan["bank"][q], verifier, r["response"]) for q, r in zip(ids, rows)]
            for q, flags_q in zip(ids, flags):
                scored.append(dict(question_id=q, order=o, style=s, checks=flags_q))
            bits[o,s] = [all(f) for f in flags]
            fractions[o,s] = [mean(f) for f in flags]
            passed, total = sum(map(sum,flags)), sum(map(len,flags))
            lengths = [r["response_tokens"] for r in rows]
            cell = dict(all_pass=sum(bits[o,s]), n=len(ids), constraints_passed=passed, constraints_total=total,
                        mean_tokens=mean(lengths), median_tokens=median(lengths))
            print(f"{o} {s}: all seven {cell['all_pass']}/{len(ids)} ({100*mean(bits[o,s]):.1f}%), "
                  f"constraints {passed}/{total} ({100*passed/total:.1f}%), mean/median tokens {mean(lengths):.1f}/{median(lengths):g}")
            if s != "neutral":
                diff = paired(list(zip(bits[o,"neutral"], bits[o,s])))
                cell["vs_neutral"] = diff
                constraint_diff = paired_mean([x-y for x,y in zip(fractions[o,s], fractions[o,"neutral"])])
                cell["constraints_vs_neutral"] = constraint_diff
                print(f"  vs neutral {diff['gain_pp']:+.2f} pp, 95% CI {diff['ci95_gain_pp']}, p={diff['mcnemar_exact_p']:.4g}")
                print(f"  constraints vs neutral {constraint_diff['gain_pp']:+.2f} pp, "
                      f"95% CI {constraint_diff['ci95_pp']}, p={constraint_diff['p']}")
            results[f"{o}/{s}"] = cell
    contrasts = {}
    for metric, values in (("all_seven", bits), ("constraints", fractions)):
        contrasts[metric] = {}
        for s in STYLES[1:]:
            differences = [(an-nn)-(ap-np) for an,nn,ap,np in zip(values["tau-1",s], values["tau-1","neutral"],
                                                                  values["tau+1",s], values["tau+1","neutral"])]
            contrasts[metric][s] = v = paired_mean(differences)
            print(f"{metric} / {s}-neutral gap, tau-1 minus tau+1: {v['gain_pp']:+.2f} pp, "
                  f"95% CI {v['ci95_pp']}, p={v['p']}")
    atomic_text(folder/"pbif-scores.jsonl", "".join(json.dumps(r)+"\n" for r in scored))
    atomic_text(folder/"summary.json", json.dumps(dict(matched=len(ids), cells=results, interactions=contrasts), indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "prepare", "summary"))
    args = parser.parse_args()
    import ifeval_carrier as flip
    pilot = flip.pilot
    folder, cache = OUTPUT, OUTPUT/"upstream"
    if (PREVIOUS/"upstream").is_dir():
        cache = PREVIOUS/"upstream"
    config = folder/"run_config.json"
    if args.mode == "summary":
        if not config.exists():
            print("PBIF is awaiting preparation.")
            return
        plan = json.loads(config.read_text())
        sources(cache)
        report(folder, plan, pilot.saved(folder, plan), registry(cache))
        return
    with pilot.job_lock(folder):
        sources(cache)
        bank = question_bank(cache)
        verifier = registry(cache, prepare=True)
        for item in bank.values():
            checks(item, verifier, "Preflight check.")
        previous_config = PREVIOUS/"run_config.json"
        previous = json.loads(previous_config.read_text()) if previous_config.exists() else None
        if config.exists():
            plan = json.loads(config.read_text())
            if plan["protocol"] != PROTOCOL or any(row["spoken_prompt"] != PHRASE for row in plan["records"]):
                raise ValueError("Frozen plan uses a different spoken instruction")
            if bank != plan["bank"] or plan["data_revision"] != HF_REV or plan["scorer_revision"] != CODE_REV:
                raise ValueError("Frozen plan differs from the pinned release")
        else:
            clips = normal_clips(folder, previous, flip)
            plan = make_plan(bank, clips, flip)
            freeze(config, plan)
        reuse_answers(folder, plan, previous, pilot)
        print(f"{len(bank)} written PBIF requests, {len(plan['records'])} planned answers.", flush=True)
        if args.mode == "prepare":
            return
        if len(pilot.saved(folder, plan, repair=True)) < len(plan["records"]):
            with pilot.gpu_queue():
                from diffa_model_receipts import model_revisions
                from src.dataloader import build_chat_prompt
                freeze(folder/"model-revisions.json", model_revisions())
                pilot.generate(folder, plan, flip.night.DiffaBackend(), build_chat_prompt)
        report(folder, plan, pilot.saved(folder, plan), verifier)


if __name__ == "__main__":
    main()
