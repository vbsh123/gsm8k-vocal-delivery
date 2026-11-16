from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moss_audio_io import load_audio


class MossAudioTests(unittest.TestCase):
    def test_full_stereo_pcm_wav_matches_upstream_mono_and_resampling(self):
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'long.wav'
            t = np.arange(40*24000)/24000
            samples = np.stack([0.2*np.sin(2*np.pi*440*t), 0.1*np.sin(2*np.pi*220*t)], axis=1)
            sf.write(path, samples, 24000, subtype='PCM_16')
            # Reference is the pinned MOSS loader's algorithm, using the older
            # TorchAudio SoundFile backend explicitly to avoid TorchCodec.
            original, sr = torchaudio.load(path, backend='soundfile')
            expected = torchaudio.functional.resample(original.mean(dim=0, keepdim=True), sr, 16000).squeeze(0).numpy()
            with patch.object(torchaudio, 'load', side_effect=AssertionError('TorchCodec path must not be used')):
                actual = load_audio(str(path), 16000)
            self.assertEqual(actual.shape, (40*16000,))
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_array_equal(actual, expected)
            self.assertGreater(np.abs(actual[-16000:]).max(), 0.01)

    def test_mono_matching_rate_is_decoded_without_resampling(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'mono.wav'
            sf.write(path, np.linspace(-0.5, 0.5, 16000), 16000, subtype='PCM_16')
            expected, _ = sf.read(path, dtype='float32')
            with patch('torchaudio.functional.resample', side_effect=AssertionError('No resampling needed')):
                actual = load_audio(str(path), 16000)
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
