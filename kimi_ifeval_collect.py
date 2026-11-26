"""Collect paired whole-response activations on the fitting half of spoken IFEval."""
import argparse
import json
from pathlib import Path

import kimi_bbh_layer_diagnostic as kimi
import kimi_spoken_ifeval as spoken
from phi4_queued_workers import gpu_queue
from phi4_sweep import job_lock
from winning_diffa_sweeps import digest, freeze

OUTPUT = Path("outputs/kimi-spoken-ifeval-answer-vectors")
STYLES = ("neutral", "angry")


def prepare():
    from evaluate_kimi_audio import MODEL, REVISION, SOURCE_REVISION, TOKENIZER_MODEL, TOKENIZER_REVISION
    split = json.loads(spoken.SPLIT.read_text())
    ids, test = split["collection_ids"], split["test_ids"]
    if (len(ids) != 254 or len(test) != 254 or len(set(ids + test)) != 508
            or split["official_revision"] != spoken.official.REVISION):
        raise ValueError("Spoken IFEval split differs from the original 254/254 split.")
    expected = dict(model=MODEL, revision=REVISION, source_revision=SOURCE_REVISION,
                    tokenizer_model=TOKENIZER_MODEL, tokenizer_revision=TOKENIZER_REVISION,
                    official_revision=spoken.official.REVISION, task="ifeval", seed=42,
                    prompt=spoken.official.PROMPT, params=dict(kimi.PARAMS, max_new_tokens=1024))
    source = spoken.OUTPUT/"run_config.json"
    saved = json.loads(source.read_text())
    if any(saved.get(k) != v for k,v in expected.items()) or set(saved["bank"]) != set(ids + test):
        raise ValueError(f"Kimi model, prompts or decoding settings differ at {source}")
    records = {}
    for r in saved["records"]:
        key = r["question_id"], r["condition"]
        if key[0] not in ids or key[1] not in STYLES:
            continue
        if (key in records or r["voice"] != "en-US-JennyNeural"
                or digest(r["audio_file"]) != r["audio_sha256"]):
            raise ValueError(f"Duplicate, changed or unexpected Jenny recording: {key}")
        records[key] = r
    plan = dict(**expected, protocol="kimi-spoken-ifeval-whole-answer-mean-v1",
                split_sha256=digest(spoken.SPLIT), source_sha256=digest(source), collection_ids=ids, test_ids=test,
                layers=list(kimi.LAYERS), token_group="answer", styles=list(STYLES),
                pooling="Mean last-position hidden state across all native next-token generation forwards "
                        "of each response. Subtract neutral per question, then average questions equally.",
                records=[records[q,s] for q in ids for s in STYLES])
    freeze(OUTPUT/"run_config.json", plan)
    return plan


def path_for(q, style):
    return OUTPUT/"captures"/f"{q}-{style}.pt"


def capture(model, layers, record, plan):
    from evaluate_kimi_audio import messages
    kimi.seed()
    with kimi.capture_means(layers) as (sums, counts):
        _, response = model.generate(messages(Path(record["audio_file"]), plan["prompt"]), **plan["params"])
    if set(counts) != set(plan["layers"]) or min(counts.values()) < 1:
        raise ValueError("Incomplete per-layer answer capture.")
    means = kimi.finish_means(sums, counts)
    return dict(answer_means=means, response=response, forwards=counts[0])


def summarize(plan):
    for style in plan["styles"]:
        count = sum(path_for(q,style).is_file() for q in plan["collection_ids"])
        print(f'{style}: {count}/{len(plan["collection_ids"])} whole-response captures', flush=True)
    print("All-layer direction file:", OUTPUT/"style-vectors.pt" if (OUTPUT/"style-vectors.pt").exists() else "pending")


def finalize(plan, receipt):
    import torch
    totals = {}
    for q in plan["collection_ids"]:
        means = {}
        for style in plan["styles"]:
            saved = torch.load(path_for(q,style), map_location="cpu", weights_only=True)
            value = saved["answer_means"]
            if ((saved["question_id"],saved["condition"],saved["config_sha256"]) != (q,style,receipt)
                    or value.shape != (len(plan["layers"]),3584) or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid cached activation at {q}/{style}")
            means[style] = value
        for style in plan["styles"][1:]:
            delta = (means[style]-means["neutral"]).double()
            totals[style] = delta if style not in totals else totals[style]+delta
    vectors = {s:(v/len(plan["collection_ids"])).float() for s,v in totals.items()}
    kimi.atomic_save(OUTPUT/"style-vectors.pt", dict(mean_differences=vectors, layers=plan["layers"],
                     group="answer", sign="emotion minus neutral", pooling=plan["pooling"],
                     config_sha256=receipt, collection_ids=plan["collection_ids"]))


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["prepare","run","summary"])
    action = p.parse_args().action
    config = OUTPUT/"run_config.json"
    if action == "summary":
        if config.is_file():
            summarize(json.loads(config.read_text()))
        else:
            print("Awaiting preparation.")
        return
    with job_lock(OUTPUT):
        plan = prepare()
        summarize(plan)
        if action == "prepare":
            return
        receipt = digest(config)
        (OUTPUT/"captures").mkdir(exist_ok=True)
        pending = [r for r in plan["records"] if not path_for(r["question_id"],r["condition"]).is_file()]
        if pending:
            with gpu_queue(), torch.inference_mode():
                model = kimi.load_model(Path("external/Kimi-Audio"))
                layers = kimi.decoder_layers(model)
                for i,r in enumerate(pending,len(plan["records"])-len(pending)+1):
                    if digest(r["audio_file"]) != r["audio_sha256"]:
                        raise ValueError(f"Recording changed: {r['audio_file']}")
                    result = capture(model, layers, r, plan)
                    kimi.atomic_save(path_for(r["question_id"],r["condition"]), dict(
                        **result, question_id=r["question_id"], condition=r["condition"], config_sha256=receipt))
                    print(f'{i}/{len(plan["records"])} {r["question_id"]} {r["condition"]}: '
                          f'{result["forwards"]} answer-generation forwards, all 28 layers', flush=True)
        finalize(plan, receipt)
        summarize(plan)


if __name__ == "__main__":
    main()
