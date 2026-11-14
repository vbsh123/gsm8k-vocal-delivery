import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


def shared_module():
    path = Path(__file__).resolve().parents[1]/'extract_vectors.py'
    spec = importlib.util.spec_from_file_location('chunked_shared_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'librosa': SimpleNamespace()}):
        spec.loader.exec_module(module)
    return module


class Inputs(SimpleNamespace):
    def to(self, device):
        self.input_features = self.input_features.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.seen = []

    def forward(self, input_features, attention_mask):
        self.seen.append(input_features.clone())
        valid = int(attention_mask.sum())
        h = input_features.transpose(1, 2)
        return SimpleNamespace(last_hidden_state=h[:, :valid], hidden_states=(h, h+10),
                               output_lengths=torch.tensor([valid]))


class Model:
    stage = 2
    def __init__(self):
        self.whisper_model = Encoder()
        self.semantic_input = None
        self.acoustic_input = None
        self.arguments = None

    def acoustic_connector(self, layers):
        self.acoustic_input = layers
        return layers[0].mean(dim=1, keepdim=True).expand(1, 64, 1)

    def semantic_connector(self, values):
        self.semantic_input = values
        return values

    def get_speech_features(self, *args, **kwargs):
        raise AssertionError('Native feature function should be bypassed for long audio')

    def generate(self, tokenizer, **kwargs):
        self.arguments = kwargs
        self.speech, self.mask = self.get_speech_features(None, None, stage=2)
        return 'Final answer: A'


def extractor(audio, **kwargs):
    # The sentinel represents Whisper's padded frames, which must never be joined.
    features = torch.full((1, 1, 6), 999.)
    features[0, 0, :len(audio)] = torch.as_tensor(audio.copy())
    mask = torch.zeros((1, 6), dtype=torch.long)
    mask[:, :len(audio)] = 1
    return Inputs(input_features=features, attention_mask=mask)


class ChunkedAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = shared_module()

    def test_both_chunks_reach_each_adapter_with_padding_removed(self):
        model = Model()
        audio = np.arange(1, 8, dtype=np.float32)
        speech, mask, info = self.shared.chunked_speech_features(audio, 16000, model, extractor, 4/16000)
        self.assertEqual(len(model.whisper_model.seen), 2)
        self.assertEqual(info['valid_encoder_frames'], [4, 3])
        self.assertEqual(info['joined_encoder_frames'], 7)
        self.assertEqual(speech.shape, (1, 71, 1))
        self.assertEqual(mask.shape, (1, 71))
        self.assertEqual(info['acoustic_tokens'], 64)
        torch.testing.assert_close(model.semantic_input.flatten(), torch.arange(1., 8.))
        torch.testing.assert_close(model.acoustic_input[0].flatten(), torch.arange(1., 8.))
        torch.testing.assert_close(model.acoustic_input[1].flatten(), torch.arange(11., 18.))
        torch.testing.assert_close(speech[:, :64], torch.full((1, 64, 1), 4.))

    def test_native_audio_delegates_directly_to_unchanged_gen(self):
        audio = np.zeros(16000, dtype=np.float32)
        with patch.object(self.shared, 'gen', return_value='same response') as native, \
             patch.object(self.shared, 'chunked_speech_features') as chunks:
            response, info = self.shared.gen_chunked(audio, 16000, 'model', 'tokenizer', 'features', 'prompt', 512)
        self.assertEqual(response, 'same response')
        self.assertEqual(info['mode'], 'native')
        self.assertEqual(info['chunks'], 1)
        native.assert_called_once()
        chunks.assert_not_called()

    def test_long_generation_keeps_decoder_settings_and_restores_encoder_method(self):
        model = Model(); original = model.get_speech_features
        response, info = self.shared.gen_chunked(np.arange(7, dtype=np.float32), 16000,
            model, 'tokenizer', extractor, 'same prompt', 512, 4/16000)
        self.assertEqual(response, 'Final answer: A')
        self.assertEqual(info['chunks'], 2)
        self.assertEqual(model.get_speech_features, original)
        self.assertEqual(model.arguments['input_texts'], ['same prompt'])
        self.assertEqual(model.arguments['accelerate'], 'fast_dllm')
        self.assertEqual(model.arguments['steps'], 128)
        self.assertEqual(model.arguments['block_length'], 32)
        self.assertEqual(model.arguments['max_new_tokens'], 512)
        self.assertEqual(model.arguments['temperature'], 0.)

    def test_encoder_method_is_restored_if_decoder_raises(self):
        model = Model(); original = model.get_speech_features
        with patch.object(model, 'generate', side_effect=RuntimeError('decoder failed')):
            with self.assertRaisesRegex(RuntimeError, 'decoder failed'):
                self.shared.gen_chunked(np.zeros(7, dtype=np.float32), 16000,
                    model, None, extractor, '', 512, 4/16000)
        self.assertEqual(model.get_speech_features, original)

    def test_invalid_chunk_length_rejected_before_native_generation(self):
        with self.assertRaises(ValueError):
            self.shared.gen_chunked(np.zeros(10), 16000, None, None, None, '', 512, 31)


if __name__ == '__main__':
    unittest.main()
