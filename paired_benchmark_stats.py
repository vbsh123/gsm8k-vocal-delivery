"""Read the saved paired scores for PBIF emotion tests or Kimi IFEval placement."""
import argparse
import json
from pathlib import Path

from rescore_audio_steering import paired
from steer import read_rows

STYLES = ("angry","happy","sad","terrified")


def pbif(model):
    folder = {"diffa":"diffa-pbif-added-styles-rate0","kimi":"kimi-pbif-five-styles"}[model]
    path = Path("outputs") / folder / "pbif-scores.jsonl"
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Generate the five-style PBIF summary on its host first.")
    from kimi_pbif_spoken import emotion_test
    config = json.loads((path.parent/"run_config.json").read_text())
    emotion_test(path.parent, config["bank"], STYLES)


def placement():
    from kimi_spoken_ifeval import OUTPUT, EXTRA_OUTPUT
    from kimi_ifeval_flip import OUTPUT as written
    def load_score(folder):
        path = folder/"official-scored.jsonl"
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Run the corresponding summary first.")
        result = {}
        for r in read_rows(path):
            key = r["question_id"],r["condition"]
            if key in result:
                raise ValueError(f"Duplicate {key} at {path}")
            result[key] = r["official_scores"]
        return result
    heard = load_score(OUTPUT)
    for key,score in load_score(EXTRA_OUTPUT).items():
        if key in heard and heard[key] != score:
            raise ValueError(f"Conflicting spoken scores for {key}")
        heard[key] = score
    flip = load_score(written)
    styles = ("neutral", *STYLES)
    common = heard.keys() & flip.keys()
    ids = sorted(q for q,s in common if s=="neutral" and all((q,style) in common for style in styles))
    print(f"Kimi written minus spoken accuracy, {len(ids)}/508 five-style matched prompts.")
    if not ids:
        return
    for style in styles:
        for mode in ("strict","loose"):
            v = paired([(heard[q,style][mode]["prompt"],flip[q,style][mode]["prompt"]) for q in ids])
            print(f"{style} {mode}: {v['baseline_correct']}/{len(ids)} -> {v['steered_correct']}/{len(ids)}, "
                  f"{v['gain_pp']:+.2f} pp, 95% CI {v['ci95_gain_pp']}, exact p={v['mcnemar_exact_p']:.4g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", choices=["diffa-pbif","kimi-pbif","kimi-placement"])
    task = parser.parse_args().comparison
    if task == "kimi-placement":
        placement()
    else:
        pbif(task.split("-")[0])
