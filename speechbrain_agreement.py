"""Compare SpeechBrain's IEMOCAP emotion classifier with DIFFA on neutral clips."""

from emotion_agreement import run


MODEL = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
LABELS = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}


def parse_prediction(labels, scores):
    scores = [float(score) for score in scores]
    index = max(range(len(scores)), key=scores.__getitem__)
    label = labels[index]
    emotion = LABELS.get(label.lower(), label.lower())
    return {"emotion": emotion, "predicted_label": label,
            "labels": labels, "scores": scores}


def load_predictor(device):
    import torch
    from speechbrain.inference.interfaces import foreign_class

    classifier = foreign_class(
        source=MODEL,
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        savedir="external/speechbrain-emotion",
        run_opts={"device": device},
    )
    classifier.eval()

    def predict(audio):
        # The shared runner supplies mono, float32 audio at 16 kHz.
        with torch.inference_mode():
            probabilities, _, _, _ = classifier.classify_batch(
                torch.from_numpy(audio).unsqueeze(0)
            )
        # This checkpoint's YAML uses ordinary softmax, despite the interface
        # docstring calling out_prob "log posterior probabilities".
        scores = probabilities.detach().cpu().reshape(-1).tolist()
        labels = [classifier.hparams.label_encoder.ind2lab[i] for i in range(len(scores))]
        return parse_prediction(labels, scores)

    return predict


if __name__ == "__main__":
    run("speechbrain", MODEL, load_predictor)
