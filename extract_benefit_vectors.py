"""Average saved angry-minus-neutral directions using a training-outcome filter."""

import argparse
import json
from pathlib import Path

from extract_audio_vectors import GROUPS, save_tensor
from scoring import score_response


def keep_pair(row, manifest, selection):
    answer = manifest[row["pair"]["neutral_id"]]["answer"]
    neutral = score_response(row["neutral_response"], answer)["correct"]
    angry = score_response(row["angry_response"], answer)["correct"]
    if selection == "exclude_losses":
        return angry or not neutral
    if selection == "angry_correct":
        return angry
    return angry and not neutral


def angry_helped(row, manifest):
    return keep_pair(row, manifest, "angry_helped")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--filter", choices=["angry_helped", "exclude_losses", "angry_correct"], default="angry_helped")
    args = p.parse_args()
    import torch

    plan = json.loads((args.collection / "collection-config.json").read_text())
    total, selected = None, []
    for index, pair in enumerate(plan["pairs"]):
        row = torch.load(args.collection / "pairs" / f"{index:05d}.pt", map_location="cpu", weights_only=True)
        if row["pair"] != pair:
            p.error("Saved pair order differs from its collection plan")
        if not keep_pair(row, plan["manifest"], args.filter):
            continue
        difference = row["difference"] if "difference" in row else row["angry"] - row["neutral"]
        if total is None:
            total = torch.zeros_like(difference)
        total += difference
        selected.append(pair)
    if not selected:
        p.error("No training pairs pass the requested filter")
    excluded = sorted({r["question_id"] for r in plan["manifest"].values()})
    method = {"angry_helped": "benefit_mean", "exclude_losses": "exclude_losses_mean",
              "angry_correct": "angry_correct_mean"}[args.filter]
    selection = {"angry_helped": "First-set angry correct / neutral incorrect; no second-set outcomes used",
                 "exclude_losses": "Exclude neutral correct / angry incorrect; retain both wrong",
                 "angry_correct": "Retain angry correct, regardless of neutral correctness"}[args.filter]
    folder = args.output / method
    folder.mkdir(parents=True, exist_ok=True)
    state = dict(pair_mean_sums={"angry": total},
                 pair_counts={"angry": torch.full((total.shape[0],), len(selected), dtype=torch.long)},
                 groups=GROUPS, styles=["angry"], prompt=plan["prompt"], max_new_tokens=plan["max_new_tokens"],
                 steps=128, seed=42, seed_policy="constant_before_every_generation",
                 sign="style minus neutral", method=method, supported_modes=["fixed"],
                 averaging=plan.get("averaging", "matched"), pairs=selected,
                 extraction_question_ids=excluded,
                 selection=selection)
    checkpoint = folder / "style-vectors.pt"
    if checkpoint.exists():
        previous = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (previous["pairs"] != selected or previous["averaging"] != state["averaging"]
                or not torch.equal(previous["pair_mean_sums"]["angry"], total)):
            p.error("Saved direction changed; use another output")
    else:
        save_tensor(torch, state, checkpoint)
    (folder / "extraction-questions.jsonl").write_text("".join(json.dumps(dict(question_id=q)) + "\n" for q in excluded))
    (folder / "selected-pairs.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(f"{len(selected)}/{len(plan['pairs'])} training pairs retained; averaging={state['averaging']}; filter={args.filter}.", flush=True)
    print("Saved:", folder / "style-vectors.pt", flush=True)


if __name__ == "__main__":
    main()
