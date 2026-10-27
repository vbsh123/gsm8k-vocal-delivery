"""Exercise duration filtering without importing the GPU/model dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class DurationFilterTests(unittest.TestCase):
    def filter(self, recordings, frames):
        source = Path(__file__).resolve().parents[1] / "evaluate.py"
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

    def test_one_long_style_excludes_entire_question_and_keeps_other_question(self):
        rows = [self.row(q, style) for q in ("q1", "q2") for style in ("neutral", "angry")]
        kept, skipped = self.filter(rows, {"q1-neutral.wav": 16000, "q1-angry.wav": 480001,
                                          "q2-neutral.wav": 480000, "q2-angry.wav": 32000})
        self.assertEqual([r["question_id"] for r in kept], ["q2", "q2"])
        self.assertEqual([r["question_id"] for r in skipped], ["q1", "q1"])
        self.assertGreater(skipped[1]["audio_seconds"], 30)

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
