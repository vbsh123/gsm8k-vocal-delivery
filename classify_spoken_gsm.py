"""Classify original Spoken GSM with DIFFA or an external emotion model; no math solving."""

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path

from emotion_agreement import agreement_summary, binary_label, print_summary, read_rows


DIFFA_MODEL = "zhoujiaming777/DIFFA-2"
# Same binary prompt and generation settings as the generated-audio classifier.
DIFFA_PROMPT = ("I want you to describe me whether the emotion the speaker conveys is neutral or not, "
               "if he has an emotion in the voice, say no, if it sounds like no emotion is being shown "
               "from the speaker say yes. ONLY REPLY 1 WORD 'YES' for neutral and 'NO' for not neutral")


def diffa_prediction(response):
    label = response.strip().lower().strip(".!?")
    prediction = {"yes": "neutral", "no": "emotional"}.get(label, "unresolved")
    return {"emotion": prediction, "binary_prediction": prediction, "response": response,
            "predicted_style": label, "prompt": DIFFA_PROMPT, "system_prompt": "",
            "max_new_tokens": 512, "decoder": "fast_dllm"}


def load_diffa():
    import torch
    from extract_vectors import load_model

    model, tokenizer, feature_extractor = load_model()

    def predict(audio):
        inputs = feature_extractor(audio, sampling_rate=16000, truncation=False,
                                   return_tensors="pt", return_attention_mask=True).to("cuda")
        with torch.inference_mode():
            response = model.generate(
                tokenizer, input_audio_features=inputs.input_features,
                input_audio_mask=inputs.attention_mask, system_prompt="",
                input_texts=[DIFFA_PROMPT], steps=128, block_length=32,
                max_new_tokens=512, temperature=0.0, accelerate="fast_dllm")
        return diffa_prediction(response)

    return predict


def summarize(path, recordings):
    predictions = {}
    counts = {}
    for model in ("diffa", "emotion2vec", "sensevoice", "speechbrain"):
        rows = read_rows(path / f"results-{model}-original.jsonl")
        if rows:
            rows = {k: r for k, r in rows.items() if k in recordings}
            predictions[model] = rows
            counts[model] = {"processed": len(rows),
                             "binary": dict(Counter(r["binary_prediction"] for r in rows.values())),
                             "emotions": dict(Counter(r["emotion"] for r in rows.values()))}
    print("\nOriginal Spoken GSM classification counts")
    print(f"{'Model':14} {'Processed':>10} {'Neutral':>10} {'Emotional':>10} {'Unresolved':>11}")
    for model, row in counts.items():
        c = row["binary"]
        print(f'{model:14} {row["processed"]:10} {c.get("neutral", 0):10} '
              f'{c.get("emotional", 0):10} {c.get("unresolved", 0):11}')
        print("  Emotion tags:", row["emotions"])
    agreements = {}
    diffa = predictions.get("diffa", {})
    for model, rows in predictions.items():
        if model == "diffa":
            continue
        shared = {k: recordings[k] for k in rows if k in diffa
                  and diffa[k]["predicted_style"] in ("yes", "no")}
        if shared:
            print(f"\n{model} versus DIFFA: {len(shared)} shared clips with valid DIFFA predictions")
            agreements[model] = agreement_summary(shared, diffa, rows)
            print_summary(agreements[model])
    report = {"selected": len(recordings), "models": counts, "agreement_with_diffa": agreements}
    (path / "classification-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("diffa", "emotion2vec", "sensevoice", "speechbrain"))
    parser.add_argument("--path", type=Path, default=Path("outputs/spoken-gsm-original"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.model == "diffa" and args.device not in ("cuda", "cuda:0"):
        parser.error("The existing DIFFA loader runs on cuda:0")
    if not (args.path / "manifest.jsonl").exists():
        parser.error("Run prepare_spoken_gsm.py first")
    recordings = read_rows(args.path / "manifest.jsonl")
    if any(r["style"] != "original" or not 0 < r["duration_seconds"] <= 30 for r in recordings.values()):
        parser.error("Expected original Spoken GSM clips within 30 seconds")
    if args.limit is not None:
        recordings = dict(list(recordings.items())[:args.limit])
    adapter = importlib.import_module(f"{args.model}_agreement") if args.model != "diffa" else None
    model_id = adapter.MODEL if adapter else DIFFA_MODEL
    output = args.path / f"results-{args.model}-original.jsonl"
    completed = read_rows(output)
    if any(r["model"] != model_id for r in completed.values()):
        parser.error("Existing predictions belong to a different model")
    pending = [r for k, r in recordings.items() if k not in completed]
    print(f"{args.model}: {len(recordings)} original clips; {len(pending)} pending.", flush=True)
    if pending and not args.summary_only:
        import librosa
        import soundfile as sf

        predict = adapter.load_predictor(args.device) if adapter else load_diffa()
        with output.open("a") as stream:
            for index, recording in enumerate(pending, 1):
                audio, rate = sf.read(args.path / recording["audio_path"], dtype="float32")
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                if rate != 16000:
                    audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
                prediction = predict(audio)
                if adapter:
                    prediction["binary_prediction"] = binary_label(prediction["emotion"])
                row = {"clip_id": recording["clip_id"], "question_id": recording["question_id"],
                       "style": "original", "model": model_id, **prediction}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f'[{index}/{len(pending)}] {row["clip_id"]}: {row["emotion"]}', flush=True)
    summarize(args.path, recordings)
    print(f"\nResults: {output}")


if __name__ == "__main__":
    main()
