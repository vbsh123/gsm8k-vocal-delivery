
import argparse
import json
from pathlib import Path
import soundfile as sf
import torch
import librosa
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperModel
from peft import LoraConfig, get_peft_model
from summarize import print_summary
from scoring import score_response

from src.configuration_DIFFA import DIFFAConfig
from src.modeling_DIFFA import DIFFAModel
from src.fast_dllm.model.modeling_llada import LLaDAModelLM

llm_id = "GSAI-ML/LLaDA-8B-Instruct"
whisper_id = "openai/whisper-large-v3"
diffa_id = "zhoujiaming777/DIFFA-2"

def load_recordings(path):
    with (path / "manifest.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--path', type=Path)
    p.add_argument('--results-file', default='results.jsonl')
    p.add_argument('--max-new-tokens', type=int, default=512)
    args = p.parse_args(argv)
    if args.max_new_tokens < 32 or args.max_new_tokens % 32 or 128 % (args.max_new_tokens // 32):
        p.error('--max-new-tokens must be a multiple of 32 whose block count divides 128.')
    prompt = ("Solve the math problem in the audio. Give concise step-by-step reasoning "
              "with only the necessary calculations. If the question asks for a total, "
              "compute that total. End with one plain-text line in exactly this format: "
              "Final answer: <number>. Put a single numeric answer on that line, "
              "without Markdown, units, or separate component counts.")
    recordings = load_recordings(args.path)
    recordings, skipped = filter_recordings(args.path, recordings)
    with (args.path / (args.results_file + ".skipped.jsonl")).open("w") as f:
        for item in skipped:
            f.write(json.dumps(item) + "\n")
    print(f"Evaluating {len(recordings)} clips; skipped {len(skipped)} clips from "
          f"{len({r['question_id'] for r in skipped})} questions with long audio.", flush=True)
    if not recordings:
        print("No questions fit the 30-second limit in every recording; model was not loaded.")
        return
    results_path = args.path / args.results_file
    processed_ids = set()
    if results_path.exists():
        with results_path.open() as f:
            processed_ids = {json.loads(line)["clip_id"] for line in f if line.strip()}
    pending = sum(r["clip_id"] not in processed_ids for r in recordings)
    print(f"{pending} clips remain; completed results will be skipped.", flush=True)
    if not pending:
        print_summary(results_path)
        return
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
        print(recording["clip_id"], response, flush=True)

        score = score_response(response, recording["answer"])
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
            **score,
        }

        with results_path.open("a") as f:
            f.write(json.dumps(result) + "\n")
        processed_ids.add(recording["clip_id"])
    print_summary(args.path / args.results_file)
if __name__ == "__main__":
    main()
