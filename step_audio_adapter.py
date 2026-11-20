"""Step-Audio-2 native text-output inference, following the pinned official example.

Template/windowing: stepfun-ai/Step-Audio2@76e272b56c3917a8d7188f18bbb5a65dfc8a0845,
stepaudio2.py and utils.py. Audio feature functions come from the pinned model's
own modeling_step_audio_2 module. No transcription, crop, TTS, or forward patch.
"""
import sys

SYSTEM = 'You are a helpful assistant.'
AUDIO_WINDOW_SAMPLES = 25*16000
TEXT_TOKEN_LIMIT = 151688


def native_inputs(tokenizer, features, path, prompt, device):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    audio=features.load_audio(path)  # Official mono/resampling helper; no max_length.
    if audio.numel()==0:raise ValueError('Empty Step input recording.')
    parts=[f'<|BOT|>system\n{SYSTEM}<|EOT|>', '<|BOT|>human\n', prompt]
    mels=[]
    for start in range(0,len(audio),AUDIO_WINDOW_SAMPLES):
        mel=features.log_mel_spectrogram(audio[start:start+AUDIO_WINDOW_SAMPLES],n_mels=128,padding=479)
        mels.append(mel)
        parts.append('<audio_start>'+'<audio_patch>'*features.compute_token_num(mel.shape[1])+'<audio_end>')
    parts.extend(['<|EOT|>','<|BOT|>assistant\n'])
    ids=torch.cat([tokenizer(text=p,return_tensors='pt',padding=True)['input_ids'] for p in parts],dim=-1).to(device)
    lengths=torch.tensor([m.shape[1]-2 for m in mels],dtype=torch.int32,device=device)
    wavs=pad_sequence([m.t() for m in mels],batch_first=True,padding_value=0).transpose(1,2).to(device)
    return dict(input_ids=ids,attention_mask=torch.ones_like(ids),wavs=wavs,wav_lens=lengths)


def decode_suffix(tokenizer,suffix,eos):
    """Exclude a terminal EOS only when present; retain last token at budget limit."""
    tokens=suffix.tolist()
    generated_count=len(tokens)
    if tokens and tokens[-1]==eos:tokens=tokens[:-1]
    # Official text-output decoding excludes audio control/code tokens.
    text=tokenizer.decode([t for t in tokens if t<TEXT_TOKEN_LIMIT])
    return text,generated_count


def make_generator(path):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
    tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=True,padding_side='right')
    model=AutoModelForCausalLM.from_pretrained(path,trust_remote_code=True,torch_dtype=torch.bfloat16).eval().cuda()
    tokenizer.eos_token='<|EOT|>'
    eos=tokenizer.convert_tokens_to_ids('<|EOT|>');model.config.eos_token_id=eos
    features=sys.modules[type(model).__module__]
    generation=GenerationConfig(max_new_tokens=512,pad_token_id=tokenizer.pad_token_id,eos_token_id=eos,
        do_sample=False,num_beams=1,repetition_penalty=1.,use_cache=True)

    def generate(row,prompt):
        inputs=native_inputs(tokenizer,features,row['audio_file'],prompt,model.device)
        context=model.config.text_config.max_position_embeddings
        if inputs['input_ids'].shape[-1]+512>context:
            raise ValueError('Step input plus generation budget exceeds native context; refusing to crop.')
        with torch.inference_mode():
            outputs=model.generate(**inputs,generation_config=generation,tokenizer=tokenizer)
        suffix=outputs[0,inputs['input_ids'].shape[-1]:]
        generate.last_trace=dict(generated_token_ids=suffix.tolist(),input_tokens=inputs['input_ids'].shape[-1],ended_with_eos=bool(len(suffix) and suffix[-1].item()==eos))
        return decode_suffix(tokenizer,suffix,eos)
    # Expose the native instance to observational collectors; no forward/cache changes.
    generate.model=model
    generate.tokenizer=tokenizer
    return generate
