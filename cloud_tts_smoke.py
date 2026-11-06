"""Generate selected question deliveries using Azure and/or ElevenLabs (no GPU needed)."""

import argparse
from html import escape
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


STYLES = ("neutral", "angry", "happy", "sad")
AZURE_EXTRA_STYLES = ("excited", "terrified", "shouting")


def azure_ssml(text, style, voice, intensity):
    content = escape(text)
    if style != "neutral":
        azure_style = "cheerful" if style == "happy" else style
        content = (f'<mstts:express-as style="{azure_style}" styledegree="{intensity}">'
                   f'{content}</mstts:express-as>')
    return ('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">'
            f'<voice name="{escape(voice, quote=True)}">{content}</voice></speak>')


def eleven_text(text, style):
    tags = {"angry": "angry", "happy": "happily", "sad": "sad"}
    return text if style == "neutral" else f"[{tags[style]}] {text}"


def request_audio(url, body, headers):
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=180) as response:
            return response.read()
    except HTTPError as error:
        raise RuntimeError(f"TTS HTTP {error.code}: {error.read().decode(errors='replace')}") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["azure", "elevenlabs", "both"], default="both")
    parser.add_argument("--text", default="Maya has twelve apples and gives away five. How many apples remain?")
    parser.add_argument("--input-jsonl", type=Path, help="Question rows with question and optional question_id/id, answer.")
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--styles", nargs="+", choices=STYLES + AZURE_EXTRA_STYLES, default=list(STYLES))
    parser.add_argument("--output", type=Path, default=Path("outputs/cloud-tts-smoke"))
    parser.add_argument("--azure-voice", default="en-US-JennyNeural")
    parser.add_argument("--azure-intensity", type=float, default=1.5)
    parser.add_argument("--eleven-model", default="eleven_v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true", help="Save the requests without calling APIs.")
    args = parser.parse_args()
    if args.provider != "azure" and any(style in AZURE_EXTRA_STYLES for style in args.styles):
        parser.error("excited, terrified, and shouting currently require --provider azure.")
    if not 0.01 <= args.azure_intensity <= 2:
        parser.error("--azure-intensity must be between 0.01 and 2.")
    questions = [{"question": args.text}]
    if args.input_jsonl:
        questions = [json.loads(line) for line in args.input_jsonl.read_text().splitlines() if line.strip()]
    if not 1 <= args.num_examples <= len(questions):
        parser.error(f"Requested {args.num_examples} questions; {len(questions)} available.")
    questions = questions[:args.num_examples]

    providers = ["azure", "elevenlabs"] if args.provider == "both" else [args.provider]
    required = {"azure": ["AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION"],
                "elevenlabs": ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]}
    if not args.prepare_only:
        missing = [name for provider in providers for name in required[provider] if not os.environ.get(name)]
        if missing:
            parser.error("Set environment variables: " + ", ".join(missing))

    args.output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for provider, question_index, question in (
            (provider, index, question) for provider in providers
            for index, question in enumerate(questions)):
        text = question["question"]
        question_id = question.get("question_id", question.get("id", f"local_{question_index:05d}"))
        prefix = f"q{question_index:05d}-" if args.input_jsonl else ""
        for style in args.styles:
            if provider == "azure":
                voice = args.azure_voice
                request = azure_ssml(text, style, voice, args.azure_intensity)
                extension = "wav"
            else:
                voice = os.environ.get("ELEVENLABS_VOICE_ID", "YOUR_VOICE_ID")
                request = {"text": eleven_text(text, style), "model_id": args.eleven_model,
                           "seed": args.seed}
                extension = "mp3"
            tasks.append({"provider": provider, "style": style, "voice": voice,
                          "question_id": question_id, "question": text, "answer": question.get("answer"),
                          "request": request,
                          "audio_path": f"{prefix}{provider}-{style}.{extension}"})
    (args.output / "plan.jsonl").write_text(
        "".join(json.dumps(task) + "\n" for task in tasks), encoding="utf-8")
    cards = "\n".join(
        f'<p><b>{escape(str(task["question_id"]))} / {task["provider"]} / {task["style"]}</b><br>'
        f'{escape(task["question"])}<br>'
        f'<audio controls preload="none" src="{task["audio_path"]}"></audio></p>' for task in tasks)
    (args.output / "listen.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Cloud TTS smoke</title>'
        f'<h1>Cloud TTS smoke</h1>{cards}', encoding="utf-8")
    if args.prepare_only:
        print(f"Prepared {len(tasks)} requests in {args.output}; no API calls made.")
        return

    for index, task in enumerate(tasks, 1):
        print(f'[{index}/{len(tasks)}] {task["question_id"]} / {task["provider"]} / {task["style"]}', flush=True)
        if task["provider"] == "azure":
            region = os.environ["AZURE_SPEECH_REGION"]
            audio = request_audio(
                f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
                task["request"].encode("utf-8"),
                {"Ocp-Apim-Subscription-Key": os.environ["AZURE_SPEECH_KEY"],
                 "Content-Type": "application/ssml+xml",
                 "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
                 "User-Agent": "gsm8k-vocal-delivery"})
        else:
            audio = request_audio(
                f'https://api.elevenlabs.io/v1/text-to-speech/{quote(task["voice"], safe="")}'
                '?output_format=mp3_44100_128', json.dumps(task["request"]).encode("utf-8"),
                {"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"})
        (args.output / task["audio_path"]).write_bytes(audio)
    print(f"Saved audio and listening page: {args.output / 'listen.html'}")


if __name__ == "__main__":
    main()
