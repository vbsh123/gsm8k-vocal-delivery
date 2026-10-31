"""Classify cropped delivery styles and copy accepted clips plus all neutrals."""

import argparse
import json
from pathlib import Path
import shutil

import soundfile as sf
import torch
import librosa
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperModel
from peft import LoraConfig, get_peft_model
from src.configuration_DIFFA import DIFFAConfig
from src.modeling_DIFFA import DIFFAModel
from src.fast_dllm.model.modeling_llada import LLaDAModelLM

llm_id = "GSAI-ML/LLaDA-8B-Instruct"
whisper_id = "openai/whisper-large-v3"
diffa_id = "zhoujiaming777/DIFFA-2"

def filter_recordings(path, recordings):
    # Read actual WAV headers, rather than trusting saved duration metadata.
    durations = {}
    long_questions = set()
    for recording in recordings:
        info = sf.info(path / recording["audio_path"])
        durations[recording["clip_id"]] = info.frames / info.samplerate
        if info.frames > 30 * info.samplerate:
            long_questions.add(recording["question_id"])
    kept, skipped = [], []
    for recording in recordings:
        if recording["question_id"] in long_questions:
            skipped.append({
                "clip_id": recording["clip_id"],
                "question_id": recording["question_id"],
                "style": recording["style"],
                "audio_seconds": durations[recording["clip_id"]],
                "reason": "question_has_recording_over_30_seconds",
            })
        else:
            kept.append(recording)
    return kept, skipped

def load_recordings(path):
    with (path / "manifest.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--path', type=Path)
    p.add_argument('--results-file', default='results.jsonl')
    p.add_argument('--max-new-tokens', type=int, default=512)
    args = p.parse_args(argv)
    recordings = load_recordings(args.path)

    prompt = ("I want you to describe me whether the emotion the speaker conveys is neutral or not, if he has an emotion in the voice, say no, if it sounds like no emotion is being shown from the speaker say yes. ONLY REPLY 1 WORD 'YES' for neutral and 'NO' for not neutral")
    recordings, skipped = filter_recordings(args.path, recordings)
    tokenizer = AutoTokenizer.from_pretrained(llm_id)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_id)
    config_path = hf_hub_download(diffa_id, "config.json")
    raw_config = json.loads(Path(config_path).read_text())
    config = DIFFAConfig(**raw_config)
    config.whisper_model_id = whisper_id
    config.prompt_size = raw_config.get("prompt_size", 64)

    model = DIFFAModel(config, tokenizer=tokenizer, stage=2)
    weights_path = hf_hub_download(diffa_id, 'pytorch_model.bin')
    state = torch.load(
        weights_path, map_location="cpu", weights_only=True, mmap=True
    )
    audio_weights = {
        key: value for key, value in state.items() if not key.startswith("llm_model.")
    }
    model.load_state_dict(audio_weights, strict=False)
    whisper = WhisperModel.from_pretrained(whisper_id, torch_dtype= torch.float32)
    model.whisper_model.load_state_dict(whisper.encoder.state_dict())

    del whisper, audio_weights
    llm = LLaDAModelLM.from_pretrained(llm_id, torch_dtype=torch.bfloat16)
    llm_weights = {
        key.removeprefix("llm_model."): value
        for key, value in state.items()
        if key.startswith("llm_model.")
    }
    lora_weights = {
        key: value
        for key, value in llm_weights.items()
        if "lora_" in key
    }

    rank = next(
        value.shape[0]
        for key, value in lora_weights.items()
        if ".lora_A." in key
    )
    targets = sorted({
        key.split(".lora_")[0].split(".")[-1]
        for key in lora_weights
    })

    llm = get_peft_model(
        llm,
        LoraConfig(
            r=rank,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    peft_weights = {
        key if key.startswith("base_model.model.")
        else "base_model.model." + key: value
        for key, value in llm_weights.items()
    }
    llm.load_state_dict(peft_weights, strict=False)

    model.llm_model = llm.merge_and_unload()
    model = model.cuda().eval()
    model.requires_grad_(False)
    filtered = args.path.parent / "filtered"
    filtered.mkdir(parents=True, exist_ok=True)
    processed_ids = set()
    results_path = args.path / args.results_file
    if results_path.exists():
        with results_path.open() as f:
            processed_ids = {json.loads(line)["clip_id"] for line in f if line.strip()}

    for recording in recordings:
        if recording["clip_id"] in processed_ids:
            continue
        audio_path = args.path / recording['audio_path']
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr = sample_rate, target_sr=16000)

        inputs = feature_extractor(
            audio, sampling_rate=16000, truncation=False,
            return_tensors="pt", return_attention_mask=True,
        ).to("cuda")
        with torch.inference_mode():
            response = model.generate(
                tokenizer,
                input_audio_features=inputs.input_features,
                input_audio_mask=inputs.attention_mask,
                system_prompt="",
                input_texts=[prompt],
                steps=128,
                block_length=32,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                accelerate="fast_dllm",
            )
        expected_style = recording["style"]
        predicted_style = response.strip().lower().strip(".!?")
        matches_requested = (predicted_style == "no" and expected_style != "neutral") or (predicted_style == "yes" and expected_style == "neutral")
        print(recording["clip_id"], response, flush=True)
        if expected_style == "neutral" or matches_requested:
            source = args.path / recording["audio_path"]
            destination = filtered / recording["audio_path"]

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

            with (filtered / "manifest.jsonl").open("a") as f:
                f.write(json.dumps(recording) + "\n")


        response_tokens = len(tokenizer.encode(response, add_special_tokens=False))
        result = {
            "clip_id": recording["clip_id"],
            "question_id": recording["question_id"],
            "style": recording["style"],
            "speaker": recording["speaker"],
            "repeat": recording["repeat"],
            "response": response,
            "prompt": prompt,
            "system_prompt": "",
            "max_new_tokens": args.max_new_tokens,
            "decoder": "fast_dllm",
            "response_tokens_reencoded": response_tokens,
            "near_token_limit": response_tokens >= args.max_new_tokens - 2,
            "predicted_style": predicted_style,
            "matches_requested": matches_requested,
        }
        processed_ids.add(recording["clip_id"])
        with (args.path / args.results_file).open("a") as f:
            f.write(json.dumps(result) + "\n")
    print("done")


main()
