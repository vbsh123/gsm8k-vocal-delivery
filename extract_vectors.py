"""Shared DIFFA-2 model loading and audio generation."""

import json
from pathlib import Path

import librosa
import torch
from diffa_model_receipts import model_revisions


def load_model():
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperModel
    from peft import LoraConfig, get_peft_model
    from src.configuration_DIFFA import DIFFAConfig
    from src.modeling_DIFFA import DIFFAModel
    from src.fast_dllm.model.modeling_llada import LLaDAModelLM

    llm_id = "GSAI-ML/LLaDA-8B-Instruct"
    whisper_id = "openai/whisper-large-v3"
    diffa_id = "zhoujiaming777/DIFFA-2"
    revisions = model_revisions()

    tokenizer = AutoTokenizer.from_pretrained(llm_id, revision=revisions.get(llm_id))
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_id, revision=revisions.get(whisper_id))
    config_path = hf_hub_download(diffa_id, "config.json", revision=revisions.get(diffa_id))
    raw_config = json.loads(Path(config_path).read_text())
    config = DIFFAConfig(**raw_config)
    config.whisper_model_id = whisper_id
    config.prompt_size = raw_config.get("prompt_size", 64)

    model = DIFFAModel(config, tokenizer=tokenizer, stage=2)
    weights_path = hf_hub_download(diffa_id, 'pytorch_model.bin', revision=revisions.get(diffa_id))
    state = torch.load(
        weights_path, map_location="cpu", weights_only=True, mmap=True
    )
    audio_weights = {
        key: value for key, value in state.items() if not key.startswith("llm_model.")
    }
    model.load_state_dict(audio_weights, strict=False)
    whisper = WhisperModel.from_pretrained(whisper_id, torch_dtype= torch.float32, revision=revisions.get(whisper_id))
    model.whisper_model.load_state_dict(whisper.encoder.state_dict())

    del whisper, audio_weights
    llm = LLaDAModelLM.from_pretrained(llm_id, torch_dtype=torch.bfloat16, revision=revisions.get(llm_id))
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


def chunked_speech_features(audio, sample_rate, model, fextractor, chunk_seconds=30):
    """Encode chunks separately, then run BOTH adapters once over the joined valid features."""
    if sample_rate != 16000 or audio.ndim != 1:
        raise ValueError('Chunk encoder expects mono 16-kHz audio.')
    if not 0 < chunk_seconds <= 30 or not len(audio):
        raise ValueError('Need nonempty audio and chunks of at most 30 seconds.')
    if model.stage != 2:
        raise ValueError('Chunked audio currently supports DIFFA dual-adapter stage=2 only.')
    size = int(chunk_seconds * sample_rate)
    device = next(model.whisper_model.parameters()).device
    last_states, layer_states, lengths = [], [], []
    with torch.inference_mode():
        for start in range(0, len(audio), size):
            inputs = fextractor(audio[start:start+size], sampling_rate=sample_rate, truncation=False,
                return_tensors='pt', return_attention_mask=True).to(device)
            encoded = model.whisper_model(input_features=inputs.input_features,
                                          attention_mask=inputs.attention_mask)
            valid = int(encoded.output_lengths[0].item())
            lengths.append(valid)
            last_states.append(encoded.last_hidden_state[:, :valid])
            layer_states.append(tuple(h[:, :valid] for h in encoded.hidden_states))
        joined_last = torch.cat(last_states, dim=1)
        joined_layers = tuple(torch.cat([chunk[layer] for chunk in layer_states], dim=1)
                              for layer in range(len(layer_states[0])))
        acoustic = model.acoustic_connector(joined_layers)
        semantic = model.semantic_connector(joined_last)
        speech = torch.cat([acoustic, semantic], dim=1)
        mask = torch.ones(speech.shape[:2], dtype=torch.bool, device=speech.device)
    info = dict(mode='chunked-whisper-global-adapters-v1', chunks=len(lengths),
        chunk_seconds=chunk_seconds, valid_encoder_frames=lengths,
        joined_encoder_frames=joined_last.shape[1], acoustic_tokens=acoustic.shape[1],
        semantic_tokens=semantic.shape[1], sample_rate=sample_rate, samples=len(audio))
    return speech, mask, info


def gen_chunked(audio, sample_rate, model, tokenizer, fextractor, prompt, max_new_tokens,
                chunk_seconds=30):
    """Experimental long-audio path. Native-size audio takes the original gen path unchanged."""
    if not 0 < chunk_seconds <= 30 or not len(audio):
        raise ValueError('Need nonempty audio and chunks of at most 30 seconds.')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    if len(audio) <= int(chunk_seconds * sample_rate):
        return gen(audio, sample_rate, model, tokenizer, fextractor, prompt, max_new_tokens), dict(
            mode='native', chunks=1, chunk_seconds=chunk_seconds, samples=len(audio), sample_rate=sample_rate)
    speech, mask, info = chunked_speech_features(audio, sample_rate, model, fextractor, chunk_seconds)
    with torch.inference_mode():
        # generate() normally calls this method to encode audio. Use the already
        # encoded full recording, leaving its prompt construction and decoder unchanged.
        original = model.get_speech_features
        model.get_speech_features = lambda *args, **kwargs: (speech, mask)
        try:
            response = model.generate(tokenizer, input_audio_features=None, input_audio_mask=None,
                system_prompt='', input_texts=[prompt], steps=128, block_length=32,
                max_new_tokens=max_new_tokens, temperature=0., accelerate='fast_dllm')
        finally:
            model.get_speech_features = original
    return response, info
