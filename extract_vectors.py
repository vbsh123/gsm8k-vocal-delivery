"""Extract per-layer delivery directions from DIFFA's first denoising pass."""

import argparse
import hashlib
import json
from pathlib import Path
import argparse
import json
from pathlib import Path
import soundfile as sf
import torch
import librosa

import src.fast_dllm.diffa_generate as decoder

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




def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--path', type=Path)
    p.add_argument('--results-file', default='results.jsonl')
    p.add_argument('--max-new-tokens', type=int, default=512)
    args = p.parse_args(argv)

    model, tokenizer, fextractor = load_model()
    path = args.path / "results-neutral-binary.jsonl"

    with path.open() as f:
        results = [json.loads(line) for line in f if line.strip()]

    passed = [result for result in results if result["style"] in ["angry","confident", "happy"] and result["predicted_style"]== "no"]
    prompt = ("Solve the math problem in the audio. Give concise step-by-step reasoning "
                "with only the necessary calculations. If the question asks for a total, "
                "compute that total. End with one plain-text line in exactly this format: "
                "Final answer: <number>. Put a single numeric answer on that line, "
                "without Markdown, units, or separate component counts.")

    def make_hook(layer_index):
        def hook(module, inputs, output):
            if len(captured[layer_index]) >= 512:
                return

            hidden = output[0]
            acoustic = hidden[0, bounds["start"]:bounds["acoustic_end"]]
            semantic = hidden[0, bounds["acoustic_end"]:bounds["end"]]
            combined = hidden[0, bounds["start"]:bounds["end"]]
            answer = hidden[0, -args.max_new_tokens:]

            summaries = torch.stack([
                acoustic.float().mean(dim=0),
                semantic.float().mean(dim=0),
                combined.float().mean(dim=0),
                answer.float().mean(dim=0)
            ])
            captured[layer_index].append(summaries.cpu())
        return hook

    layers = model.llm_model.model.transformer.blocks

    for i in range(len(layers)):
        layer = model.llm_model.model.transformer.blocks[i]
        layer.register_forward_hook(make_hook(i))
    
    with (args.path / "manifest.jsonl").open() as f:
        recordings = {
            row["clip_id"]: row
            for line in f if line.strip()
            for row in [json.loads(line)]
        }
    neutrals = {
        (r["question_id"], r["speaker"], r["repeat"]): r
        for r in recordings.values()
        if r["style"] == "neutral"
    }
    bounds = {}
    
    def capture_insert(text_embeds, speech_embeds, audio_info_lengths):
        start = int(audio_info_lengths[0].item())

        bounds["start"] = start
        bounds["acoustic_end"] = start + model.config.prompt_size
        bounds["end"] = start + speech_embeds.shape[1]

        return original_insert(
            text_embeds, speech_embeds, audio_info_lengths
    )
    
    original_insert = decoder.insert_audio_embedding
    decoder.insert_audio_embedding = capture_insert
    styles = ["angry", "happy", "confident"]
    groups = ["acoustic", "semantic", "combined", "answer"]
    max_passes = 512
    hidden_size = model.llm_model.get_input_embeddings().weight.shape[1]
    sums = {
    s: torch.zeros(len(layers), max_passes, len(groups), hidden_size)
    for s in styles
    }
    counts = {
        s: torch.zeros(len(layers), max_passes, dtype=torch.long)
        for s in styles
    }
    pair_mean_sums = {
        s: torch.zeros(len(layers), len(groups), hidden_size)
        for s in styles
    }
    pair_counts = {
        s: torch.zeros(len(layers), dtype=torch.long)
        for s in styles
    }
    for passedResult in passed:
        recording = recordings[passedResult["clip_id"]]

        key = (
            recording["question_id"],
            recording["speaker"],
            recording["repeat"],
        )
        neutral = neutrals[key]

        audio_path = args.path / recording['audio_path']
        audio, sample_rate = sf.read(audio_path, dtype="float32")
   
        neutral_path = args.path / neutral['audio_path']
        neutral_audio, neutral_sample_rate = sf.read(neutral_path, dtype="float32")

        style = recording["style"]

        
        
        captured = [[] for _ in layers]
        styled_response = gen(audio,sample_rate,model,tokenizer,fextractor,prompt,args.max_new_tokens)
        styled_captured = captured
        captured = [[] for _ in layers]
        bounds = {}
        neutral_response = gen(neutral_audio,neutral_sample_rate,model,tokenizer,fextractor,prompt,args.max_new_tokens)
        neutral_captured = captured
        print(recording["clip_id"], styled_response, neutral_response, flush=True)
        for layer_index in range(len(layers)):
            n = min(
                len(styled_captured[layer_index]),
                len(neutral_captured[layer_index]),
            )
            if n == 0:
                continue

            styled = torch.stack(styled_captured[layer_index][:n])
            neutral = torch.stack(neutral_captured[layer_index][:n])

            difference = styled - neutral

            sums[style][layer_index, :n] += difference
            counts[style][layer_index, :n] += 1

            pair_mean_sums[style][layer_index] += difference.mean(dim=0)
            pair_counts[style][layer_index] += 1

    output_path = args.path.parent / "style_vectors.pt"

    torch.save({
        "sums": sums,
        "counts": counts,
        "pair_mean_sums": pair_mean_sums,
        "pair_counts": pair_counts,
        "groups": groups,
        "styles": styles,
        "max_passes": max_passes,
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "sign": "style minus neutral",
    }, output_path)

    print("Saved:", output_path)

        

if __name__ == "__main__":
    main()
