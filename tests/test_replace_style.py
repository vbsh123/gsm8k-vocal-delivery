import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from generate import read_jsonl, write_json, write_jsonl
from replace_style import replace_style


class ReplacementTests(unittest.TestCase):
    def make_dataset(self, path, styles, version):
        path.mkdir()
        tasks, records = [], []
        for style in styles:
            clip_id = f"question__Ryan__{style}__r00"
            task = dict(clip_id=clip_id, question_id="question", text="Two plus two?",
                        answer="4", source_answer="4", speaker="Ryan", repeat=0,
                        seed=42, style=style, instruction=f"{style} {version}",
                        audio_path=f"audio/{clip_id}.wav")
            tasks.append(task)
            records.append(dict(task, duration_seconds=1.0))
            wav = path / task["audio_path"]
            wav.parent.mkdir(exist_ok=True)
            wav.write_bytes((f"{style}-{version}" * 20).encode())
        write_json(path / "run_config.json", dict(tasks=tasks, model="model", model_revision="main",
                   temperature=0.9, max_new_tokens=2048, language="English", wav_subtype="FLOAT"))
        write_jsonl(path / "plan.jsonl", tasks)
        write_jsonl(path / "manifest.jsonl", records)

    def test_replaces_only_selected_style_and_archives_stale_results(self):
        with tempfile.TemporaryDirectory() as directory:
            target, source = Path(directory) / "original", Path(directory) / "replacement"
            self.make_dataset(target, ["neutral", "angry", "stressed"], "v1")
            self.make_dataset(source, ["angry"], "v2")
            write_jsonl(target / "results.jsonl", [{"correct": True}])
            angry = "audio/question__Ryan__angry__r00.wav"
            neutral = "audio/question__Ryan__neutral__r00.wav"
            old_angry, old_neutral = (target / angry).read_bytes(), (target / neutral).read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                backup = replace_style(target, source, "angry")
            self.assertEqual((target / angry).read_bytes(), (source / angry).read_bytes())
            self.assertEqual((backup / angry).read_bytes(), old_angry)
            self.assertEqual((target / neutral).read_bytes(), old_neutral)
            self.assertTrue((backup / "results.jsonl").exists())
            self.assertFalse((target / "results.jsonl").exists())
            manifest = read_jsonl(target / "manifest.jsonl")
            self.assertEqual(len(manifest), 3)
            self.assertEqual(manifest[1]["instruction"], "angry v2")
            config = json.loads((target / "run_config.json").read_text())
            self.assertEqual(config["tasks"][1]["instruction"], "angry v2")
            self.assertIn("<audio controls", (target / "listen.html").read_text())

    def test_incomplete_replacement_is_rejected_before_target_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            target, source = Path(directory) / "original", Path(directory) / "replacement"
            self.make_dataset(target, ["angry"], "v1")
            self.make_dataset(source, ["angry"], "v2")
            (source / "manifest.jsonl").unlink()
            before = (target / "run_config.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "not complete"):
                replace_style(target, source, "angry")
            self.assertEqual((target / "run_config.json").read_bytes(), before)
            self.assertFalse((target / "backups").exists())
