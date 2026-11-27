"""Collect paired DIFFA spoken/flipped PBIF directions without applying steering."""
import argparse
import csv
import io
import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import torch

import diffa_pbif_spoken as spoken
import pbif_emotions as pbif
from diffa_model_receipts import model_revisions
from extract_audio_vectors import save_tensor
from kimi_pbif import BUNDLE
from pbif_steering import split_requests
from phi4_queued_workers import gpu_queue
from phi4_sweep import job_lock
from winning_diffa_sweeps import atomic_text, digest, freeze

OUT = Path("outputs/diffa-pbif-placement-vectors")
PLACEMENTS = ("spoken", "flipped")
GROUPS = ("initial-acoustic", "initial-answer", "answer")
BUDGET = 1024
CARRIER_HASHES = {
    "neutral": "ccd4fc8029efb43bc2ecb2c35260492ff0818c14ef9894e7269284a88c6e36e0",
    "angry": "f3acd1aa30850f8951c07c9793637bd189d85d25bdb004a8f0e4e44991643bcb",
}


def prepare(folder):
    bundle = next((p for p in [Path("/workspace")/BUNDLE, Path(BUNDLE), Path("outputs")/BUNDLE]
                   if p.is_file()), None)
    cache = folder/"upstream"
    clips = {}
    if bundle:
        with ZipFile(bundle) as archive:
            hashes = {f"upstream/{name}": checksum for name, (_, checksum) in pbif.SOURCES.items()}
            hashes.update({f"{style}.wav": checksum for style, checksum in CARRIER_HASHES.items()})
            for name, checksum in hashes.items():
                data = archive.read(name)
                if sha256(data).hexdigest() != checksum:
                    raise ValueError(f"Unexpected PBIF input in bundle: {name}")
                path = folder/name
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists() or digest(path) != checksum:
                    path.write_bytes(data)
            for style, checksum in CARRIER_HASHES.items():
                clips[0, style] = dict(voice="en-US-JennyNeural", audio_file=str((folder/f"{style}.wav").resolve()),
                                       audio_sha256=checksum)
    else:
        config = pbif.OUTPUT/"run_config.json"
        if not config.is_file():
            raise FileNotFoundError(f"Place {BUNDLE} in /workspace for the two original flipped carrier WAVs.")
        clips = pbif.normal_clips(pbif.OUTPUT, json.loads(config.read_text()), None)
        if any(clips[0,s]["audio_sha256"] != checksum for s,checksum in CARRIER_HASHES.items()):
            raise ValueError("Flipped inputs differ from the original carriers.")
        cache = next((d/"upstream" for d in (pbif.PREVIOUS, pbif.OUTPUT) if (d/"upstream").is_dir()), cache)
    pbif.sources(cache)
    bank = pbif.question_bank(cache)
    fit, held = split_requests(bank)
    records = {"spoken": spoken.recordings(bank), "flipped": pbif.question_records(bank, clips, pbif.STYLES)}
    records = {k:[r for r in values if r["question_id"] in fit] for k, values in records.items()}
    expected = {(q,o,s) for q in fit for o in pbif.ORDERS for s in pbif.STYLES}
    for items in records.values():
        if len(items) != len(expected) or {(r["question_id"],r["order"],r["style"]) for r in items} != expected:
            raise ValueError("Missing or duplicate fitting recording")
    plan = dict(protocol="diffa-pbif-placement-vectors-v1", collection_ids=fit, test_ids=held,
                max_new_tokens=BUDGET, groups=list(GROUPS), layers=list(range(32)), records=records,
                model_revisions=model_revisions(), data_revision=pbif.HF_REV)
    freeze(folder/"run_config.json", plan)
    return plan


def capture(model, decoder, generate):
    # First-forward acoustic prefix and answer mask, plus the full answer buffer on every forward.
    layers = model.llm_model.model.transformer.blocks
    if len(layers) != 32:
        raise ValueError("Expected 32 DIFFA language blocks.")
    original = decoder.insert_audio_embedding
    bounds = {}
    acoustic, steps, handles = [], [], []
    current = []

    def insert(text, speech, lengths):
        start = int(lengths[0].item())
        bounds.update(start=start, acoustic_end=start+model.config.prompt_size, end=start+speech.shape[1])
        return original(text, speech, lengths)

    def observe(index):
        def hook(module, args, result):
            h = result[0].detach()
            if index != len(current) or h.shape[0] != 1 or h.shape[1] < BUDGET:
                raise ValueError("Unexpected decoder order or incomplete answer buffer.")
            if not steps:
                if not bounds or not 0 <= bounds["start"] < bounds["acoustic_end"] <= bounds["end"] <= h.shape[1]-BUDGET:
                    raise ValueError("Unexpected initial acoustic span.")
                acoustic.append(h[0,bounds["start"]:bounds["acoustic_end"]].mean(0, dtype=torch.float32))
            current.append(h[0,-BUDGET:].mean(0, dtype=torch.float32))
            if index == len(layers)-1:
                steps.append(torch.stack(current).cpu())
                current.clear()
        return hook

    try:
        decoder.insert_audio_embedding = insert
        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(observe(i)))
        generate()
    finally:
        decoder.insert_audio_embedding = original
        for handle in handles:
            handle.remove()
    if not steps or current:
        raise ValueError("Incomplete decoder capture.")
    return torch.stack(acoustic).cpu(), torch.stack(steps)


def pair_delta(neutral, angry):
    n_acoustic, n_answer = neutral
    a_acoustic, a_answer = angry
    count = min(len(n_answer), len(a_answer))
    # Match chronological forwards separately for each order, as in the L31 runner.
    delta = torch.stack((a_acoustic-n_acoustic, a_answer[0]-n_answer[0],
                         (a_answer[:count]-n_answer[:count]).mean(0)), dim=1)
    if not torch.isfinite(delta).all():
        raise ValueError("Nonfinite activation difference")
    return delta, count


def pair_path(folder, q, placement, order):
    return folder/"pairs"/f"{q}-{placement}-{order}.pt"


def collect(folder, plan, pending, receipt):
    from extract_vectors import load_model
    import diffa_happy_nightly as natural
    import src.fast_dllm.diffa_generate as decoder
    lookup = {(p,r["question_id"],r["order"],r["style"]):r for p,rows in plan["records"].items() for r in rows}
    with gpu_queue():
        model, tokenizer, features = load_model()
        freeze(folder/"runtime.json", dict(model_revisions=model_revisions(), decoder_sha256=digest(decoder.__file__)))
        for i, (q,p,o) in enumerate(pending, 1):
            traces = []
            for s in pbif.STYLES:
                r = lookup[p,q,o,s]
                traces.append(capture(model, decoder, lambda: natural.generate(r, plan, r["prompt"], model, tokenizer, features)))
            delta, count = pair_delta(*traces)
            save_tensor(torch, dict(delta=delta, forwards=[len(t[1]) for t in traces], matched=count,
                                    key=[q,p,o], config_sha256=receipt), pair_path(folder,q,p,o))
            del traces
            print(f"{4*len(plan['collection_ids'])-len(pending)+i}/{4*len(plan['collection_ids'])} pairs: {q} {p} {o}, {count} matched forwards.", flush=True)


def report(folder, plan, receipt, *, export=False):
    ids = plan["collection_ids"]
    complete = [q for q in ids if all(pair_path(folder,q,p,o).is_file() for p in PLACEMENTS for o in pbif.ORDERS)]
    print(f"{len(list((folder/'pairs').glob('*.pt')))}/{4*len(ids)} pairs saved; {len(complete)}/{len(ids)} requests matched across placements and orders.")
    if not complete:
        return
    sums = {}
    for q in complete:
        fold = ids.index(q) % 2
        for p in PLACEMENTS:
            orders = []
            for o in pbif.ORDERS:
                item = torch.load(pair_path(folder,q,p,o), map_location="cpu", weights_only=True)
                if item["config_sha256"] != receipt or item["key"] != [q,p,o]:
                    raise ValueError(f"Incompatible saved pair {q} {p} {o}")
                orders.append(item["delta"])
            value = torch.stack(orders).double().mean(0)
            if (p,fold) not in sums:
                sums[p,fold] = torch.zeros_like(value)
            sums[p,fold] += value
    total = {p:sum(sums[p,f] for f in (0,1) if (p,f) in sums) for p in PLACEMENTS}
    if export and len(complete) == len(ids):
        for p in PLACEMENTS:
            (folder/p).mkdir(exist_ok=True)
            save_tensor(torch, dict(pair_mean_sums={"angry":total[p].float()},
                        pair_counts={"angry":torch.full((32,),len(ids))}, extraction_question_ids=ids,
                        groups=list(GROUPS), config_sha256=receipt, max_new_tokens=BUDGET,
                        method="mean", mode="fixed", placement=p), folder/p/"style-vectors.pt")
    def cosine(a,b):
        den = a.norm(dim=-1)*b.norm(dim=-1)
        return (a*b).sum(-1)/den.masked_fill(den == 0, float("nan"))
    sp, fl = (total[p]/len(complete) for p in PLACEMENTS)
    columns = dict(spoken_norm=sp.norm(dim=-1), flipped_norm=fl.norm(dim=-1), cosine=cosine(sp,fl))
    for title, a, b in (("spoken_A_flipped_B",("spoken",0),("flipped",1)),
                        ("spoken_B_flipped_A",("spoken",1),("flipped",0)),
                        ("spoken_split_half",("spoken",0),("spoken",1)),
                        ("flipped_split_half",("flipped",0),("flipped",1))):
        columns[title] = cosine(sums[a],sums[b]) if a in sums and b in sums else torch.full_like(columns["cosine"], float("nan"))
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["layer","group","matched_requests",*columns])
    for i in plan["layers"]:
        for g,group in enumerate(GROUPS):
            writer.writerow([i,group,len(complete),*[v[i,g].item() for v in columns.values()]])
            if i in (0,15,31):
                print(f"L{i:02} {group:16} cosine {columns['cosine'][i,g]:+.3f}  spoken/flipped norms {sp[i,g].norm():.3f}/{fl[i,g].norm():.3f}")
    name = "layer-comparison.csv" if len(complete)==len(ids) else "interim-layer-comparison.csv"
    atomic_text(folder/name,buf.getvalue())
    print(folder/name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run","prepare","summary"))
    action = parser.parse_args().action
    config = OUT/"run_config.json"
    if action == "summary":
        if config.is_file():
            report(OUT, json.loads(config.read_text()), digest(config))
        else:
            print("Awaiting preparation.")
        return
    with job_lock(OUT):
        plan = prepare(OUT)
        receipt = digest(config)
        (OUT/"pairs").mkdir(exist_ok=True)
        pending = [(q,p,o) for q in plan["collection_ids"] for p in PLACEMENTS for o in pbif.ORDERS
                   if not pair_path(OUT,q,p,o).exists()]
        print(f"{len(pending)} neutral/angry pairs pending.", flush=True)
        if action == "run":
            if pending:
                collect(OUT,plan,pending,receipt)
            report(OUT,plan,receipt,export=True)


if __name__ == "__main__":
    main()
