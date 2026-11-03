"""Compare emotion2vec+ Large with saved DIFFA labels on neutral clips."""

from emotion_agreement import run


MODEL = "emotion2vec/emotion2vec_plus_large"


def parse_prediction(result):
    scores = [float(score) for score in result["scores"]]
    labels = result["labels"]
    index = max(range(len(scores)), key=scores.__getitem__)
    # Official labels can be bilingual, e.g. "中立/neutral".
    emotion = labels[index].split("/")[-1].strip().lower()
    return {"emotion": emotion, "predicted_label": labels[index],
            "labels": labels, "scores": scores}


def load_predictor(device):
    from funasr import AutoModel

    model = AutoModel(model=MODEL, hub="hf", device=device, disable_update=True)

    def predict(audio):
        result = model.generate(input=audio, granularity="utterance", extract_embedding=False)
        return parse_prediction(result[0])

    return predict


if __name__ == "__main__":
    run("emotion2vec", MODEL, load_predictor)
