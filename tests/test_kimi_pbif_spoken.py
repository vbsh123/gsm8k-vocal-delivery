"""Regression tests for full-utterance Kimi prompt validation."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import wave

import torch

from kimi_pbif_spoken import check_history, long_audio_guard


class LongAudioTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "speech.wav"
        with wave.open(str(self.path), "wb") as dst:
            dst.setparams((1, 2, 16000, 0, "NONE", ""))
            dst.writeframes(bytes(16000 * 2 * 30) + b"\x01\x00" * 6400)
        self.plan = dict(records=[dict(seconds=30.4, audio_file=str(self.path))],
                         prompt="Follow the instructions in the audio.", params=dict(max_new_tokens=1024))

    def history(self, chats, output_type="text"):
        with wave.open(str(chats[-1]["content"])) as src:
            samples = torch.frombuffer(bytearray(src.readframes(src.getnframes())), dtype=torch.int16)
        ids = samples[::1280].long().unsqueeze(0)
        mask = torch.ones_like(ids, dtype=torch.bool)
        return SimpleNamespace(to_tensor=lambda: (ids, None, mask, None, None),
                               continuous_feature=[ids.unsqueeze(-1).float()])

    def test_checks_real_prompt_path_and_detects_a_truncated_stream(self):
        model = SimpleNamespace(prompt_manager=SimpleNamespace(get_prompt=self.history),
                                alm=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8192)))
        chats = [dict(message_type="audio", content=str(self.path))]
        with long_audio_guard(model, self.plan):
            history = model.prompt_manager.get_prompt(chats)
        self.assertEqual(history.continuous_feature[0].shape[1], 380)
        with self.assertRaisesRegex(ValueError, "exceeds Kimi context"):
            check_history(history, self.path, 1000, 1024)
        history.continuous_feature[0] = history.continuous_feature[0][:, :375]
        with self.assertRaisesRegex(ValueError, "expected 380.*Whisper 375"):
            check_history(history, self.path, 8192, 1024)

    def test_rejects_unresponsive_tail_even_when_lengths_match(self):
        def fixed(chats, **kwargs):
            return self.history([dict(content=str(self.path))])
        model = SimpleNamespace(prompt_manager=SimpleNamespace(get_prompt=fixed),
                                alm=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8192)))
        with self.assertRaisesRegex(ValueError, "tail unchanged"):
            with long_audio_guard(model, self.plan):
                pass
