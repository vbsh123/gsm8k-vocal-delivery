"""Decode our WAV recordings without TorchCodec; retain MOSS mono/resampling steps."""


def load_audio(path: str, sample_rate: int):
    import soundfile as sf
    import torch
    from torchaudio.functional import resample

    samples, original_sample_rate = sf.read(path, dtype='float32', always_2d=True)
    if not len(samples) or sample_rate <= 0:
        raise ValueError('Expected nonempty audio and a positive target sample rate.')
    waveform = torch.from_numpy(samples.T.copy())
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if original_sample_rate != sample_rate:
        waveform = resample(waveform, orig_freq=original_sample_rate, new_freq=sample_rate)
    return waveform.squeeze(0).cpu().numpy()
