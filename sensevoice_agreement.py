"""Compare SenseVoice Small with saved DIFFA labels on neutral clips."""

import re

from emotion_agreement import EMOTIONS, run


MODEL = "FunAudioLLM/SenseVoiceSmall"


def parse_prediction(result):
    text = result["text"]
    tokens = re.findall(r"<\|([^|]+)\|>", text)
    emotions = [token.lower() for token in tokens
                if token.lower() in EMOTIONS | {"neutral", "emo_unknown"}]
    emotion = emotions[0] if len(emotions) == 1 else "unknown"
    return {"emotion": emotion, "raw_text": text}


def load_predictor(device):
    from funasr import AutoModel

    # No VAD: obtain one emotion prediction for the same complete cropped clip.
    model = AutoModel(model=MODEL, hub="hf", device=device, disable_update=True)

    def predict(audio):
        result = model.generate(input=audio, cache={}, language="en", use_itn=False)
        return parse_prediction(result[0])

    return predict


if __name__ == "__main__":
    run("sensevoice", MODEL, load_predictor)
