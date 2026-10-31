"""Exercise duration filtering without importing the GPU/model dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class DurationFilterTests(unittest.TestCase):
    source_name = "evaluate.py"

    def filter(self, recordings, frames):
        source = Path(__file__).resolve().parents[1] / self.source_name
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "filter_recordings")
        namespace = {"sf": SimpleNamespace(info=Mock(side_effect=lambda path:
                     SimpleNamespace(frames=frames[path.name], samplerate=16000)))}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        return namespace["filter_recordings"](Path("recordings"), recordings)

    def row(self, question, style):
        clip = f"{question}-{style}"
        return dict(clip_id=clip, question_id=question, style=style, audio_path=f"{clip}.wav",
                    duration_seconds=1)  # Deliberately stale metadata.

    def test_long_style_does_not_exclude_its_neutral_or_other_question(self):
        rows = [self.row(q, style) for q in ("q1", "q2") for style in ("neutral", "angry")]
        kept, skipped = self.filter(rows, {"q1-neutral.wav": 16000, "q1-angry.wav": 480001,
                                          "q2-neutral.wav": 480000, "q2-angry.wav": 32000})
        self.assertEqual([r["clip_id"] for r in kept], ["q1-neutral", "q2-neutral", "q2-angry"])
        self.assertEqual([r["clip_id"] for r in skipped], ["q1-angry"])
        self.assertGreater(skipped[0]["audio_seconds"], 30)

    def test_long_neutral_excludes_short_style(self):
        rows = [self.row("q1", style) for style in ("neutral", "angry")]
        kept, skipped = self.filter(rows, {"q1-neutral.wav": 480001, "q1-angry.wav": 16000})
        self.assertEqual(kept, [])
        self.assertEqual(len(skipped), 2)
        self.assertTrue(all(r["reason"] == "neutral_recording_over_30_seconds" for r in skipped))

    def test_missing_neutral_is_skipped(self):
        rows = [self.row("q1", "angry")]
        kept, skipped = self.filter(rows, {"q1-angry.wav": 16000})
        self.assertEqual(kept, [])
        self.assertEqual(skipped[0]["reason"], "missing_neutral_recording")

    def test_every_question_can_be_excluded(self):
        rows = [self.row("q1", "neutral")]
        kept, skipped = self.filter(rows, {"q1-neutral.wav": 640000})
        self.assertEqual(kept, [])
        self.assertEqual(len(skipped), 1)

    def test_exactly_thirty_seconds_is_kept(self):
        rows = [self.row("q1", "neutral")]
        kept, skipped = self.filter(rows, {"q1-neutral.wav": 480000})
        self.assertEqual(kept, rows)
        self.assertEqual(skipped, [])


class EmotionDurationFilterTests(DurationFilterTests):
    source_name = "filter.py"
