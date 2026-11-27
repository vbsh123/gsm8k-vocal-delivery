"""Fit a DIFFA angry-minus-neutral answer vector and steer neutral PBIF requests."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random

import pbif_emotions as pbif
from extract_audio_vectors import save_tensor
from phi4_queued_workers import gpu_queue
from phi4_sweep import job_lock
from steer import install_interventions
from winning_diffa_sweeps import digest, freeze, read_vector

OUTPUT = Path("outputs/diffa-pbif-angry-x1")
LABEL = "neutral-L31-answer-x1"
LAYER = 31
SPOKEN_OUTPUT = Path("outputs/diffa-pbif-spoken-angry-x1")


def make_plan(base, clips, bank, revisions, baseline_hash):
    import ifeval_carrier as flip
    canonical = pbif.make_plan(bank, clips, flip)
    if any(base[k] != v for k, v in canonical.items()):
        raise ValueError("Saved PBIF inputs or generation recipe differ from the original normal-rate run.")
    return steering_plan(base, bank, revisions, baseline_hash, "diffa-pbif-angry-x1-full-answer-v1")


def split_requests(bank):
    ids = sorted(bank)
    if len(ids) != 198:
        raise ValueError("Expected the original 198 distinct requests.")
    random.Random(42).shuffle(ids)
    return ids[:99], ids[99:]


def steering_plan(base, bank, revisions, baseline_hash, protocol):
    fit, test = split_requests(bank)
    neutral = [dict(r, condition=f"{r['order']}/{LABEL}", steering=LABEL)
               for r in base["records"] if r["style"] == "neutral" and r["question_id"] in test]
    return dict(protocol=protocol, baseline_sha256=baseline_hash,
                collection_ids=fit, test_ids=test, model_revisions=revisions, layer=LAYER, group="answer",
                strength=1., averaging="Full answer buffer per forward, matched forwards per order, "
                "equal order weight within each request and equal request weight.",
                max_new_tokens=base["max_new_tokens"], recipe=base["recipe"], records=neutral)


def report(plan, original, added, verifier, bank, folder):
    held = plan["test_ids"]
    # Compare all three conditions on exactly the steered rows available for each order.
    rows = {}
    for q, condition in added:
        order = added[q, condition]["order"]
        controls = [(q, f"{order}/{s}") for s in pbif.STYLES]
        if any(key not in original for key in controls):
            continue
        for key in controls:
            rows[key] = original[key]
        rows[q, condition] = added[q, condition]
    print(f"Steered {len(added)}/{len(plan['records'])} held-out neutral generations.", flush=True)
    if rows:
        subset = dict(bank={q:bank[q] for q in held}, records=[None]*(len(held)*6))
        pbif.report(folder, subset, rows, verifier, (*pbif.STYLES, LABEL))


def main():
    import ifeval_carrier as flip
    from diffa_model_receipts import model_revisions
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["run", "summary", "prepare"])
    p.add_argument("--spoken", action="store_true", help="Fit on full spoken requests and steer held-out neutral speech.")
    args = p.parse_args()
    action = args.action
    import diffa_pbif_spoken as spoken
    base_dir = spoken.OUTPUT if args.spoken else pbif.OUTPUT
    folder = SPOKEN_OUTPUT if args.spoken else OUTPUT
    source = base_dir/"run_config.json"
    config = folder/"run_config.json"
    if action == "summary" and not config.exists():
        print(f"{folder.name} is awaiting preparation.")
        return
    editing = action != "summary"
    with (job_lock(folder) if editing else nullcontext()), (job_lock(base_dir) if args.spoken and editing else nullcontext()):
        cache = next((d/"upstream" for d in (pbif.PREVIOUS, pbif.OUTPUT, base_dir)
                      if (d/"upstream").is_dir()), base_dir/"upstream")
        pbif.sources(cache)
        bank = pbif.question_bank(cache)
        verifier = pbif.registry(cache, prepare=editing)
        if args.spoken and editing:
            recipe = dict(flip.night.vanilla.RECIPE, max_new_tokens=1024)
            base = spoken.baseline_plan(bank, spoken.recordings(bank), recipe)
            freeze(source, base)
        else:
            base = json.loads(source.read_text())
        saved = flip.pilot.saved(base_dir, base, repair=args.spoken and action == "run")
        if not args.spoken and len(saved) != 792:
            raise ValueError(f"Expected the completed PBIF neutral/angry run in {base_dir}; found {len(saved)}/792.")
        if action == "summary":
            plan = json.loads(config.read_text())
            if plan["baseline_sha256"] != digest(source):
                raise ValueError("Saved PBIF baseline configuration changed.")
        else:
            revision_file = base_dir/"model-revisions.json"
            if args.spoken and not saved:
                freeze(revision_file, model_revisions())
            revisions = json.loads(revision_file.read_text())
            if revisions != model_revisions():
                raise ValueError("Model revision differs from the PBIF baseline.")
            if args.spoken:
                plan = steering_plan(base, bank, revisions, digest(source), "diffa-spoken-pbif-angry-x1-full-answer-v1")
            else:
                clips = pbif.normal_clips(base_dir, base, flip)
                plan = make_plan(base, clips, bank, revisions, digest(source))
            freeze(config, plan)
        held = set(plan["test_ids"])
        missing_controls = False
        if args.spoken:
            control_count = sum(q in held for q, _ in saved)
            missing_controls = control_count < 4 * len(held)
            print(f"Spoken held-out natural neutral/angry controls: {control_count}/{4 * len(held)}.")
        vec = folder/"style-vectors.pt"
        fit_state = folder/"fit.pt"
        if fit_state.exists():
            import torch
            state = torch.load(fit_state, map_location="cpu", weights_only=True)
            print(f"Fitting: {len(state['ids'])}/99 requests saved.", flush=True)
        else:
            print("Fitting: 0/99 requests saved.", flush=True)
        new = flip.pilot.saved(folder, plan, repair=action == "run")
        if action == "run" and (missing_controls or len(new) < len(plan["records"])):
            from src.dataloader import build_chat_prompt
            with gpu_queue():
                backend = flip.night.DiffaBackend()
                def generate(record):
                    prompt = record["prompt"]
                    flip.pilot.evaluation.context_guard(backend.tokenizer, prompt, build_chat_prompt)
                    return backend.generate(record, dict(plan, prompt=prompt), capture=False)
                layers = backend.model.llm_model.model.transformer.blocks
                inputs = {(r["question_id"], r["condition"]):r for r in base["records"]}
                fit_direction(plan, inputs, generate, layers[LAYER], fit_state, vec)
                if missing_controls:
                    print("Generating missing held-out natural controls.", flush=True)
                    flip.pilot.generate(base_dir, base, backend, build_chat_prompt, question_ids=held)
                    saved = flip.pilot.saved(base_dir, base)
                steer(plan, new, generate, layers, vec)
            new = flip.pilot.saved(folder, plan)
        if action != "prepare":
            report(plan, saved, new, verifier, bank, folder)
        else:
            print("Prepared 99 fitting and 99 held-out requests, each in both orders.")


def capture(layer, generate, record, budget):
    import torch
    trace = []
    def observe(module, inputs, output):
        hidden = output[0]
        if hidden.shape[0] != 1 or hidden.shape[1] < budget:
            raise ValueError("Expected the full answer buffer in the decoder output.")
        trace.append(hidden[0, -budget:].detach().float().mean(0).cpu())
    handle = layer.register_forward_hook(observe)
    try:
        generate(record)
    finally:
        handle.remove()
    if not trace:
        raise ValueError("No L31 forwards captured.")
    return torch.stack(trace)


def fit_direction(plan, original, generate, layer, checkpoint, vector):
    import torch
    ids = plan["collection_ids"]
    receipt = digest(checkpoint.parent/"run_config.json")
    state = dict(ids=[], total=None, config_sha256=receipt)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["config_sha256"] != receipt or state["ids"] != ids[:len(state["ids"])]:
            raise ValueError("Fit checkpoint differs from the frozen plan.")
    for q in ids[len(state["ids"]):]:
        differences = []
        for order in pbif.ORDERS:
            traces = [capture(layer, generate, original[q, f"{order}/{s}"], plan["max_new_tokens"])
                      for s in pbif.STYLES]
            neutral, angry = traces
            steps = min(len(neutral), len(angry))
            differences.append((angry[:steps] - neutral[:steps]).mean(0))
        delta = torch.stack(differences).double().mean(0)
        if not torch.isfinite(delta).all():
            raise ValueError(f"Nonfinite paired difference at {q}")
        state["total"] = delta if state["total"] is None else state["total"] + delta
        state["ids"].append(q)
        save_tensor(torch, state, checkpoint)
        print(f"Fit {len(state['ids'])}/{len(ids)} {q} (both orders)", flush=True)
    if not vector.exists():
        sums = torch.zeros(32, 1, state["total"].numel())
        sums[LAYER, 0] = state["total"].float()
        counts = torch.zeros(32, dtype=torch.long)
        counts[LAYER] = len(ids)
        save_tensor(torch, dict(pair_mean_sums={"angry":sums}, pair_counts={"angry":counts},
                                groups=["answer"], extraction_question_ids=ids, max_new_tokens=plan["max_new_tokens"],
                                mode="fixed", method="mean", config_sha256=receipt, averaging=plan["averaging"]), vector)
    direction, info = read_vector(vector, "angry")
    if (info["extraction_ids"] != ids or info["pairs"] != len(ids)
            or not torch.equal(direction, state["total"].float()/len(ids))):
        raise ValueError("Saved vector differs from the completed fit.")


def steer(plan, previous, generate, layers, vector_file):
    import torch
    direction, _ = read_vector(vector_file, "angry")
    target = next(layers[LAYER].parameters())
    direction = direction.to(device=target.device, dtype=torch.bfloat16)
    receipt = digest(vector_file.parent/"run_config.json")
    vector_hash = digest(vector_file)
    if any(r["vector_sha256"] != vector_hash for r in previous.values()):
        raise ValueError("Completed generations used a different direction.")
    count = len(previous)
    with (vector_file.parent/"deliveries.jsonl").open("a") as stream:
        for record in plan["records"]:
            if (record["question_id"], record["condition"]) in previous:
                continue
            with install_interventions(layers, {(LAYER,"answer"):direction},
                    [dict(layer=LAYER, group="answer", strength=1.)], {}, plan["max_new_tokens"]) as hooks:
                response = generate(record)
            forwards = hooks[0]["state"]["steered"]
            if not forwards:
                raise ValueError("No steered L31 forwards.")
            row = dict(record, **response, run_config_sha256=receipt, vector_sha256=vector_hash,
                       model="diffa", steered_forwards=forwards, **plan["recipe"])
            stream.write(json.dumps(row)+"\n")
            stream.flush()
            count += 1
            print(f"Steered {count}/{len(plan['records'])} {row['question_id']} {row['order']}, "
                  f"{row['response_tokens']} tokens", flush=True)


if __name__ == "__main__":
    main()
