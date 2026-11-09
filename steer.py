"""Shared steering hooks and paired reports; the historical CLI is archived."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random

from scoring import score_response


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_neutrals(records, excluded, count, seed):
    # One Ryan/repeat-0 neutral per question. Sampling never uses correctness.
    questions = {}
    for r in records:
        if (r["style"] == "neutral" and r["speaker"] == "Ryan" and r["repeat"] == 0
                and r["question_id"] not in excluded and r["duration_seconds"] <= 30):
            questions[r["question_id"]] = r
    selected = [questions[q] for q in sorted(questions)]
    random.Random(seed).shuffle(selected)
    if len(selected) < count:
        raise ValueError(f"Need {count} unseen short neutrals; only {len(selected)} available. "
                         "Crop more new recordings or lower --num-examples.")
    return selected[:count]


def make_conditions(layers, groups, strengths):
    conditions = [dict(name="baseline", layer=None, group=None, strength=0.0)]
    for layer in layers:
        for group in groups:
            for strength in strengths:
                if strength != 0:
                    conditions.append(dict(name=f"L{layer}-{group}-a{strength:g}",
                                           layer=layer, group=group, strength=strength))
    names = [c["name"] for c in conditions]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate layers, groups or strengths")
    return conditions


def reuse_questions(previous, records, excluded, count):
    """Keep exactly a previous run's questions and order for a new layer sweep."""
    if len(previous) != count:
        raise ValueError(f"Previous plan has {len(previous)} questions; requested {count}")
    for r in previous:
        if records.get(r["clip_id"]) != r or r["question_id"] in excluded:
            raise ValueError("A previous question changed or is now excluded")
    return previous


def make_steering_hook(direction, valid_passes, mode, group, strength, bounds, answer_tokens):
    """Modify only one chosen decoder block's output; count its forward calls."""
    state = {"forwards": 0, "steered": 0}

    def hook(module, inputs, output):
        step = state["forwards"]
        state["forwards"] += 1
        if strength == 0:
            return None
        if mode == "per-pass":
            if step >= len(valid_passes) or not bool(valid_passes[step]):
                return None  # No measured direction: leave this pass unchanged.
            vector = direction[step]
        else:
            vector = direction
        if group == "answer":
            positions = slice(-answer_tokens, None)
        elif group == "acoustic":
            positions = slice(bounds["start"], bounds["acoustic_end"])
        elif group == "semantic":
            positions = slice(bounds["acoustic_end"], bounds["end"])
        else:
            positions = slice(bounds["start"], bounds["end"])
        hidden = output[0].clone()
        hidden[:, positions, :] += strength * vector
        state["steered"] += 1
        return (hidden, *output[1:])

    return hook, state


def summarize(results, conditions):
    latest = {(r["clip_id"], r["condition"]): r for r in results}
    baseline = {cid: r for (cid, name), r in latest.items() if name == "baseline"}
    print("\nPaired results against baseline (parse failures count as incorrect)", flush=True)
    print(f"{'Condition':<27} {'Pairs':>5} {'Steered':>9} {'Baseline':>9} {'Wins':>5} {'Losses':>6} {'Exact p':>9}", flush=True)
    summary = []
    for condition in conditions[1:]:
        name = condition["name"]
        paired = [(r, baseline[cid]) for (cid, key), r in latest.items()
                  if key == name and cid in baseline]
        if not paired:
            continue
        n = len(paired)
        correct = sum(r["correct"] for r, _ in paired)
        base_correct = sum(b["correct"] for _, b in paired)
        wins = sum(r["correct"] and not b["correct"] for r, b in paired)
        losses = sum(b["correct"] and not r["correct"] for r, b in paired)
        disagreements = wins + losses
        p = (min(1.0, 2 * sum(math.comb(disagreements, k)
                             for k in range(min(wins, losses) + 1)) / 2**disagreements)
             if disagreements else 1.0)
        row = dict(condition=name, pairs=n, correct=correct, baseline_correct=base_correct,
                   accuracy=correct/n, baseline_accuracy=base_correct/n,
                   wins=wins, losses=losses, disagreements=disagreements,
                   mcnemar_exact_p=p)
        summary.append(row)
        print(f"{name:<27} {n:>5} {100*correct/n:>8.1f}% {100*base_correct/n:>8.1f}% "
              f"{wins:>5} {losses:>6} {p:>9.4f}", flush=True)
    print("P-values are unadjusted; a layer/strength sweep is exploratory.", flush=True)
    return summary


@contextmanager
def install_interventions(layers, directions, interventions, bounds, answer_tokens):
    """Keep all hooks for a combination active during the SAME generation."""
    handles, states = [], []
    targets = interventions or [dict(layer=0, group="answer", strength=0)]
    try:
        for target in targets:
            vector = directions[target["layer"], target["group"]] if target["strength"] else None
            hook, state = make_steering_hook(vector, None, "fixed", target["group"],
                                             target["strength"], bounds, answer_tokens)
            handles.append(layers[target["layer"]].register_forward_hook(hook))
            states.append(dict(target=target, state=state))
        yield states
    finally:
        for handle in handles:
            handle.remove()


def main():
    from archive.steering.steer import main as historical_main
    historical_main()


if __name__ == "__main__":
    main()
