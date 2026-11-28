"""Evaluate DIFFA flipped and Kimi spoken on the full IFBench test split."""
import argparse
import json
import os
from pathlib import Path
from shutil import copyfile

import run_diffa_spoken_prompt as pilot
import run_spoken_ifbench as benchmark
from cloud_tts_smoke import azure_ssml
from kimi_spoken_ifeval import generate as kimi_generate, report
from run_reasoning_transfer import generate_audio, source_records
from steer import read_rows
from winning_diffa_sweeps import digest, freeze

STYLES = ("neutral", "angry")
FOLDERS = {"diffa": Path("outputs/diffa-ifbench-flipped"), "kimi": Path("outputs/kimi-ifbench-spoken")}
VOICE = "en-US-JennyNeural"


def spoken_clips(folder, bank):
    questions = {q: dict(question_id=q, question=item["prompt"], answer="") for q, item in bank.items()}
    audio = folder/"audio"
    audio.mkdir(parents=True, exist_ok=True)
    previous = Path(os.environ.get("IFBENCH_AUDIO_DIR", "outputs/azure-spoken-ifbench"))
    # Only reuse a WAV when the source manifest has the same text, voice and SSML.
    indices = {qid: i for i, qid in enumerate(questions)}
    for row in read_rows(previous/"plan.jsonl"):
        qid, style = row["question_id"], row["style"]
        if qid not in questions or row["provider"] != "azure" or style not in STYLES:
            continue
        text = questions[qid]["question"]
        if (row["voice"], row["question"], row["request"]) != (VOICE, text, azure_ssml(text, style, VOICE, 1.5)):
            continue
        src = previous/row["audio_path"]
        dest = audio/f"q{indices[qid]:05d}-azure-{style}.wav"
        if src.is_file() and not dest.exists():
            tmp = dest.with_suffix(".part")
            copyfile(src, tmp)
            tmp.replace(dest)
    generate_audio(audio, list(questions.values()), STYLES)
    return source_records(audio, questions, STYLES)


def make_plan(model, folder, bank):
    plan = dict(protocol=f"{folder.name}-v1", name=folder.name, task="ifbench", bank=bank,
                data_revision=benchmark.REVISION, data_sha256=benchmark.DATA_SHA256,
                verifier_hashes={p: digest(benchmark.UPSTREAM/p) for p in benchmark.FILES},
                seed=42, max_new_tokens=1024, primary_metric="loose_prompt")
    if model == "diffa":
        from collie_placement import flipped_carriers
        from pbif_emotions import PHRASE
        clips = flipped_carriers(folder)
        plan.update(model="diffa", recipe=dict(pilot.evaluation.night.vanilla.RECIPE, max_new_tokens=1024))
        plan["records"] = [dict(clips[s], question_id=q, condition=s, style=s, voice=VOICE,
                                prompt=item["prompt"], spoken_prompt=PHRASE, carrier_rate=0)
                           for q, item in bank.items() for s in STYLES]
    else:
        from evaluate_kimi_audio import MODEL, REVISION, SOURCE_REVISION, TOKENIZER_MODEL, TOKENIZER_REVISION
        from kimi_bbh_layer_diagnostic import PARAMS
        clips = spoken_clips(folder, bank)
        plan.update(model=MODEL, revision=REVISION, source_revision=SOURCE_REVISION,
                    tokenizer_model=TOKENIZER_MODEL, tokenizer_revision=TOKENIZER_REVISION,
                    params=dict(PARAMS, max_new_tokens=1024), prompt=benchmark.PROMPT)
        plan["records"] = [dict(clips[q,s], condition=s, spoken_prompt=item["prompt"])
                           for q, item in bank.items() for s in STYLES]
    return plan


def summarize(folder, plan, evaluator):
    report(plan, evaluator, folder, styles=STYLES, label=plan["name"],
           primary_metric=plan["primary_metric"], scorer_revision=benchmark.REVISION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=(*FOLDERS, "audio"))
    parser.add_argument("action", choices=("run", "summary"))
    args = parser.parse_args()
    folder = FOLDERS["kimi" if args.model == "audio" else args.model]
    if args.model == "audio" and args.action == "summary":
        print(f"{sum(1 for _ in (folder/'audio').glob('q*-azure-*.wav'))}/600 IFBench recordings saved.")
        return
    config = folder/"run_config.json"
    if args.action == "summary" and not config.exists():
        print(f"{folder.name}: awaiting preparation (300 prompts, 600 answers planned).")
        if args.model == "kimi":
            print(f"{sum(1 for _ in (folder/'audio').glob('*.wav'))}/600 recordings ready.")
        return
    examples = benchmark.setup_official()
    bank = {f'ifbench_{item["key"]}': item for item in examples}
    if len(bank) != 300 or len(examples) != 300:
        raise ValueError("Expected 300 distinct official IFBench test prompts.")
    if args.model == "audio":
        with pilot.job_lock(folder):
            clips = spoken_clips(folder, bank)
            print(f"{len(clips)}/600 IFBench recordings ready in {folder/'audio'}", flush=True)
        return
    evaluator = benchmark.official_scorer()
    if args.action == "summary":
        plan = json.loads(config.read_text())
        if plan["bank"] != bank or any(digest(benchmark.UPSTREAM/p) != h for p,h in plan["verifier_hashes"].items()):
            raise ValueError("Official data or scorer differs from the frozen run.")
        return summarize(folder, plan, evaluator)
    with pilot.job_lock(folder):
        for item in examples:
            benchmark.score(item, "Preflight check.", evaluator)
        plan = make_plan(args.model, folder, bank)
        freeze(config, plan)
        done = pilot.saved(folder, plan, repair=True)
        print(f'{folder.name}: {len(done)}/600 answers saved; full 300-prompt split.', flush=True)
        if len(done) < len(plan["records"]):
            if args.model == "kimi":
                from kimi_pbif_spoken import long_audio_guard
                kimi_generate(plan, folder, input_guard=long_audio_guard)
            else:
                from diffa_model_receipts import model_revisions
                with pilot.gpu_queue():
                    from src.dataloader import build_chat_prompt
                    freeze(folder/"model-revisions.json", model_revisions())
                    pilot.generate(folder, plan, pilot.evaluation.night.DiffaBackend(), build_chat_prompt)
        summarize(folder, plan, evaluator)


if __name__ == "__main__":
    main()
