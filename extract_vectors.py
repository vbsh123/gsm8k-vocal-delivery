"""Shared DIFFA-2 model loading and audio generation."""

import json
from pathlib import Path

import librosa
import torch

def load_model():
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperModel
    from peft import LoraConfig, get_peft_model
    from src.configuration_DIFFA import DIFFAConfig
    from src.modeling_DIFFA import DIFFAModel
    from src.fast_dllm.model.modeling_llada import LLaDAModelLM

    llm_id = "GSAI-ML/LLaDA-8B-Instruct"
    whisper_id = "openai/whisper-large-v3"
    diffa_id = "zhoujiaming777/DIFFA-2"

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


    return model, tokenizer, feature_extractor


def gen(audio,sample_rate, model, tokenizer, fextractor, prompt, max_new_tokens):
    if audio.ndim == 2:
            audio = audio.mean(axis=1)

    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr = sample_rate, target_sr=16000)

    inputs = fextractor(
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
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            accelerate="fast_dllm",
            )
    return response
