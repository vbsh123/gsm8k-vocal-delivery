"""Paired DIFFA evaluation of spoken, zero-shot COLLIE requests on the Phi worker."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
from statistics import mean, median
from time import sleep

from rescore_audio_steering import download, paired
from run_spoken_ifeval import PROMPT

OUT = Path("outputs/diffa-collie-spoken")
REV = "efce1c882e995cca65e1244545167a95dba72df9"
FILES = {
    "collie/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "collie/constraints.py": "78dfee7049ee738cef6b63b665bb0f05786096e8989dd9f11d429c6736073736",
    "data/all_data.dill": "d8cf6e5447eea9457d7691c89a773ce5b512f3da1417f3480497997cab541009",
}
STYLES = ("neutral", "angry")
VECTOR_REPORT = Path("outputs/diffa-pbif-placement-vectors/layer-comparison.csv")


def release(root):
    import dill
    import nltk

    for name, checksum in FILES.items():
        path = root / name
        data = path.read_bytes() if path.exists() else download(
            f"https://raw.githubusercontent.com/princeton-nlp/Collie/{REV}/{name}")
        if sha256(data).hexdigest() != checksum:
            raise ValueError(f"COLLIE release checksum mismatch: {name}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
    try:
        nltk.word_tokenize("A sentence.")
    except LookupError as exc:
        resource = "punkt_tab" if "punkt_tab" in str(exc) else "punkt"
        nltk.download(resource, raise_on_error=True)
        nltk.word_tokenize("A sentence.")
    sys.path.insert(0, str(root.resolve()))
    with (root / "data/all_data.dill").open("rb") as source:
        return dill.load(source)


def select(dataset):
    by_family = defaultdict(dict)
    for group in sorted(dataset):
        family = group.split("_")[1]
        for i, item in enumerate(dataset[group]):
            by_family[family].setdefault(item["prompt"], (f"{group}_{i:03d}", family, item))
    rng = random.Random(42)
    selected = []
    for group in sorted(by_family):
        pool = list(by_family[group].values())
        selected.extend(rng.sample(pool, min(16, len(pool))))
    rng.shuffle(selected)
    if len(by_family) != 13 or len({row[2]["prompt"] for row in selected}) != 190:
        raise ValueError("Unexpected COLLIE cohort")
    return selected


def prepare(selected):
    import ifeval_carrier as runner
    from run_reasoning_transfer import generate_audio, source_records

    questions = {}
    for qid, family, item in selected:
        if not item["constraint"].check(item["example"], item["targets"]):
            raise ValueError(f"The released reference fails its own checker: {qid}")
        questions[qid] = dict(question_id=qid, question=item["prompt"], answer="")
    audio = OUT / "audio"
    generate_audio(audio, list(questions.values()), STYLES)
    clips = source_records(audio, questions, STYLES)
    records = [dict(clips[qid, style], condition=style, family=family, prompt=PROMPT,
                    spoken_prompt=item["prompt"]) for qid, family, item in selected for style in STYLES]
    recipe = dict(runner.night.vanilla.RECIPE, max_new_tokens=1024)
    plan = dict(protocol="diffa-collie-spoken-v1", name="Spoken COLLIE", task="collie", model="diffa",
                seed=42, revision=REV, max_new_tokens=1024, recipe=recipe, records=records)
    runner.winners.freeze(OUT / "run_config.json", plan)
    return plan


def report(plan, selected, rows, folder=OUT):
    from winning_diffa_sweeps import atomic_text

    scored = []
    scores = []
    families = defaultdict(list)
    for qid, family, item in selected:
        if not all((qid, style) in rows for style in STYLES):
            continue
        pair = tuple(bool(item["constraint"].check(rows[qid, s]["response"], item["targets"])) for s in STYLES)
        scored.append(dict(question_id=qid, family=family, **dict(zip(STYLES, pair))))
        scores.append(pair)
        families[family].append(pair)
    print(f"{plan['name']}: {len(rows)}/{len(plan['records'])} answers, "
          f"{len(scores)}/{len(selected)} matched problems.")
    if scores:
        for style in STYLES:
            lengths = [rows[r["question_id"], style]["response_tokens"] for r in scored]
            print(f"{style}: mean/median tokens {mean(lengths):.1f}/{median(lengths):g}")
        stats = paired(scores)
        print(f"Constraints satisfied: neutral {stats['baseline_correct']}/{len(scores)}, "
              f"angry {stats['steered_correct']}/{len(scores)}, {stats['gain_pp']:+.2f} pp, "
              f"95% CI {stats['ci95_gain_pp']}, exact McNemar p={stats['mcnemar_exact_p']:.4g}.")
        print("Task family   Matched   Neutral pass   Angry pass")
        for family, values in sorted(families.items()):
            n, a = map(sum, zip(*values))
            print(f"{family:12} {len(values):7} {n:14} {a:12}")
        if len(scores) == len(selected):
            atomic_text(folder/"scores.jsonl", "".join(json.dumps(r)+"\n" for r in scored))
            atomic_text(folder/"summary.json", json.dumps(stats, indent=2)+"\n")
    return scored


def main():
    import ifeval_carrier as runner
    from diffa_model_receipts import model_revisions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "summary"))
    action = parser.parse_args().action
    config = OUT / "run_config.json"
    if action == "summary":
        if not config.is_file():
            print(f"Spoken COLLIE: {sum(1 for _ in (OUT/'audio').glob('*.wav'))}/380 recordings.")
            return
        plan = json.loads(config.read_text())
        report(plan, select(release(OUT/"upstream")), runner.pilot.saved(OUT, plan))
        return
    with runner.pilot.job_lock(OUT):
        selected = select(release(OUT/"upstream"))
        plan = prepare(selected)
        if len(runner.pilot.saved(OUT, plan, repair=True)) < len(plan["records"]):
            if not VECTOR_REPORT.is_file():
                print(f"Audio ready. Waiting for the completed PBIF vector report at {VECTOR_REPORT}.", flush=True)
            while not VECTOR_REPORT.is_file():
                sleep(30)
            with runner.pilot.gpu_queue():
                from src.dataloader import build_chat_prompt
                runner.winners.freeze(OUT/"model-revisions.json", model_revisions())
                runner.pilot.generate(OUT, plan, runner.night.DiffaBackend(), build_chat_prompt)
        report(plan, selected, runner.pilot.saved(OUT, plan))


if __name__ == "__main__":
    main()
