"""Check filtering and resume behavior without loading GPU dependencies."""

import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


STYLES = ["neutral", "confident", "hesitant", "happy", "angry", "sad", "stressed"]


class FilterTests(unittest.TestCase):
    def test_partial_batches_resume_and_keep_all_neutral_baselines(self):
        source = Path(__file__).resolve().parents[1] / "filter.py"
        tree = ast.parse(source.read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        main = next(n for n in functions if n.name == "main")
        # Replace only weight loading; exercise the actual classification/save loop.
        start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "tokenizer" for t in n.targets))
        end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Expr)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                   and n.value.func.attr == "requires_grad_")
        main.body[start:end + 1] = ast.parse(
            "tokenizer = tokenizer_stub\nfeature_extractor = feature_stub\nmodel = model_stub"
        ).body
        ast.fix_missing_locations(main)

        inputs = SimpleNamespace(input_features=None, attention_mask=None)
        inputs.to = lambda device: inputs
        model = SimpleNamespace(generate=Mock(side_effect=[
            "sad", "confident", "neutral", "happy", "angry", "neutral", "stressed",
        ]))
        durations = {}
        namespace = {
            "argparse": argparse, "json": json, "Path": Path, "shutil": shutil,
            "sf": SimpleNamespace(
                info=lambda path: SimpleNamespace(frames=durations[path.name] * 16000, samplerate=16000),
                read=lambda *a, **k: (SimpleNamespace(ndim=1), 16000),
            ),
            "torch": SimpleNamespace(inference_mode=contextlib.nullcontext),
            "model_stub": model,
            "tokenizer_stub": SimpleNamespace(encode=lambda text, **kwargs: text.split()),
            "feature_stub": lambda *args, **kwargs: inputs,
        }
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)

        with tempfile.TemporaryDirectory() as tmp:
            cropped = Path(tmp) / "cropped"
            (cropped / "audio").mkdir(parents=True)
            recordings = []

            def add(question, style, duration=10):
                clip = f"{question}-{style}"
                row = dict(clip_id=clip, question_id=question, style=style,
                           speaker="Ryan", repeat=0, audio_path=f"audio/{clip}.wav")
                recordings.append(row)
                durations[f"{clip}.wav"] = duration
                (cropped / row["audio_path"]).write_bytes(clip.encode())

            for style in STYLES:
                add("q1", style)
            for style in STYLES[:-1]:
                add("q2", style)

            def run():
                (cropped / "manifest.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in recordings)
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    namespace["main"](["--path", str(cropped)])

            run()
            filtered = cropped.parent / "filtered"
            accepted = [json.loads(line) for line in (filtered / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual({r["style"] for r in accepted},
                             {"neutral", "confident", "happy", "angry", "stressed"})
            results = [json.loads(line) for line in (cropped / "emotion-results.jsonl").read_text().splitlines()]
            self.assertEqual(len(results), 7)
            self.assertFalse(results[0]["matches_requested"])
            for row in accepted:
                self.assertEqual((filtered / row["audio_path"]).read_bytes(),
                                 (cropped / row["audio_path"]).read_bytes())

            # A late overlong seventh style must not leave earlier q2 clips accepted.
            add("q2", STYLES[-1], duration=31)
            run()
            self.assertEqual(model.generate.call_count, 7)
            self.assertEqual(len((filtered / "manifest.jsonl").read_text().splitlines()), 5)
            skipped = [json.loads(line) for line in (cropped / "emotion-results.jsonl.skipped.jsonl").read_text().splitlines()]
            self.assertEqual(len(skipped), 7)
            self.assertTrue(all(r["reason"] == "question_has_recording_over_30_seconds" for r in skipped))


if __name__ == "__main__":
    unittest.main()
