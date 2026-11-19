"""Native audio inference for three independent BBH candidate model environments.

No transcription cascade, custom feature concatenation, steering, or audio crops.
Imports of model libraries are delayed until the chosen model is actually run.
"""
import json
import math
import os
from pathlib import Path
import shutil

CATALOG = Path(__file__).with_name('bbh_candidate_models.json')
MODELS = json.loads(CATALOG.read_text())
ALLOW = ['*.json', '*.py', '*.safetensors', '*.model', '*.tiktoken', '*.txt', '*.jinja', '*.jinja2', '*.yaml']
IGNORE = ['think/*', 'assets/*', 'figures/*', 'examples/*', 'static/*', 'token2wav/*']


def load_audio(path, sample_rate=16000):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    if not len(audio) or rate <= 0:
        raise ValueError('Expected a nonempty recording.')
    audio = audio.mean(axis=1)
    if rate != sample_rate:
        divisor = math.gcd(rate, sample_rate)
        audio = resample_poly(audio, sample_rate//divisor, rate//divisor)
    if not np.isfinite(audio).all():
        raise ValueError('Audio contains nonfinite samples.')
    return np.asarray(audio, dtype=np.float32)


def checkpoint(key):
    from huggingface_hub import snapshot_download
    spec = MODELS[key]
    root = Path(os.environ.get('HF_HUB_CACHE', os.environ.get('HUGGINGFACE_HUB_CACHE',
        str(Path(os.environ.get('HF_HOME', '~/.cache/huggingface')).expanduser()/'hub'))))
    root.mkdir(parents=True, exist_ok=True)
    cached = root/('models--'+spec['model'].replace('/', '--'))/'snapshots'/spec['revision']
    # A pinned snapshot already used for inference needs no weight-download reserve.
    # Verify its indexed shards, rather than mistaking a partial snapshot for a hit.
    index = cached/'model.safetensors.index.json'
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text())['weight_map'].values())
            core = ['config.json', 'tokenizer_config.json', 'tokenizer.json']
            if key != 'step': core.append('generation_config.json')
            if shards and all((cached/name).is_file() and (cached/name).stat().st_size > 0 for name in [*core, *shards]):
                print(f'{key}: reusing indexed pinned checkpoint; no weight download required.', flush=True)
                return str(cached)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    present = sum(p.stat().st_size for p in cached.rglob('*.safetensors') if p.is_file())
    needed = max(0, spec['weights_gib']*2**30-present)+4*2**30
    free = shutil.disk_usage(root).free
    if free < needed:
        raise RuntimeError(f'{key}: need approximately {needed/2**30:.1f} GiB free for uncached weights plus reserve; '
                           f'{free/2**30:.1f} GiB available. No results or caches have been deleted.')
    return snapshot_download(spec['model'], revision=spec['revision'], cache_dir=str(root),
        allow_patterns=ALLOW, ignore_patterns=IGNORE)


def phi_prompt(prompt):
    return '<|user|><|audio_1|>'+prompt+'<|end|><|assistant|>'


def make_generator(key):
    import torch
    from transformers import AutoProcessor
    path = checkpoint(key)
    if key == 'step':
        from step_audio_adapter import make_generator as step_generator
        return step_generator(path)
    if key == 'phi4':
        from transformers import AutoModelForCausalLM, GenerationConfig
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map='cuda:0', _attn_implementation='sdpa').eval()
        config = GenerationConfig.from_pretrained(path)

        def generate(task, prompt):
            audio = load_audio(task['audio_file'])
            inputs = processor(text=phi_prompt(prompt), audios=[(audio, 16000)], return_tensors='pt').to(model.device, dtype=torch.bfloat16)
            with torch.inference_mode():
                ids = model.generate(**inputs, generation_config=config, max_new_tokens=512,
                    do_sample=False, num_beams=1, repetition_penalty=1., use_cache=True)
            suffix = ids[:, inputs['input_ids'].shape[1]:]
            return processor.batch_decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0], int(suffix.shape[1])

    elif key == 'minicpm':
        from transformers import AutoModel
        model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa', init_vision=False, init_audio=True, init_tts=False).eval().cuda()
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

        def generate(task, prompt):
            audio = load_audio(task['audio_file'])
            with torch.inference_mode():
                response = model.chat(msgs=[{'role': 'user', 'content': [prompt, audio]}], processor=processor,
                    do_sample=False, max_new_tokens=512, min_new_tokens=0, num_beams=1,
                    repetition_penalty=1., max_inp_length=32768, enable_thinking=False,
                    use_tts_template=True, generate_audio=False, stream=False)
            if not isinstance(response, str):
                raise TypeError('Expected a text-only response from native MiniCPM chat.')
            # Native chat returns decoded text; this is a retokenized response length.
            return response, len(processor.tokenizer.encode(response, add_special_tokens=False))

    elif key == 'flamingo':
        from transformers import AudioFlamingo3ForConditionalGeneration
        processor = AutoProcessor.from_pretrained(path)
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(path,
            torch_dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa').eval()

        def generate(task, prompt):
            audio = load_audio(task['audio_file'])
            if len(audio) > 600*16000:
                raise ValueError('AF3 audio exceeds its native 600-second limit; refusing to truncate.')
            conversation = [{'role': 'user', 'content': [
                {'type': 'text', 'text': prompt}, {'type': 'audio', 'path': task['audio_file']}]}]
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            # Supply full decoded audio directly to the official windowing processor.
            inputs = processor(text=text, audio=[audio], sampling_rate=16000, return_tensors='pt').to(model.device, dtype=torch.bfloat16)
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                                     num_beams=1, repetition_penalty=1., use_cache=True)
            suffix = ids[:, inputs['input_ids'].shape[1]:]
            return processor.batch_decode(suffix, skip_special_tokens=True)[0], int(suffix.shape[1])
    else:
        raise ValueError(f'Unknown model: {key}')
    return generate
