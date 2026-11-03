"""Check original-audio preparation, shared duration bounds and resumable classification."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from classify_spoken_gsm import diffa_prediction, main, summarize
from prepare_spoken_gsm import export_rows


class SpokenClassificationTests(unittest.TestCase):
    def test_batch_300_runs_external_models_only_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = Path(__file__).resolve().parents[1] / "run_spoken_classification.sh"
            shutil.copy2(script, root / script.name)
            log = root / "calls.txt"
            for environment in (".venv-diffa", ".venv-emotion"):
                executable = root / environment / "bin/python"
                executable.parent.mkdir(parents=True)
                executable.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "' + str(log) + '"\n')
                executable.chmod(0o755)
            subprocess.run(["bash", str(root / script.name), "300"], check=True, capture_output=True)
            calls = log.read_text().splitlines()
            self.assertEqual(len(calls), 4)
            self.assertIn("prepare_spoken_gsm.py --num-examples 300", calls[0])
            self.assertTrue(all("--limit 300" in call for call in calls[1:]))
            self.assertEqual([call.split("--model ")[1].split()[0] for call in calls[1:]],
                             ["emotion2vec", "sensevoice", "speechbrain"])
            self.assertNotIn("--model diffa", log.read_text())

    def test_original_source_ids_skip_long_audio_without_reindexing_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = [("data/test-0.parquet", i, {"audio": {"bytes": payload}, "answer": "18"})
                    for i, payload in enumerate([b"first", b"long", b"third"])]
            def info(stream):
                return SimpleNamespace(frames=31*16000 if stream.read() == b"long" else 16000,
                                       samplerate=16000)
            writes = []
            def write(path, audio, rate, subtype):
                writes.append((audio, rate, subtype))
                Path(path).write_bytes(audio)
            fake_sf = SimpleNamespace(info=info,
                                      read=lambda stream, **kwargs: (stream.read(), 16000), write=write)
            with patch.dict(sys.modules, {"soundfile": fake_sf}), redirect_stdout(io.StringIO()):
                self.assertEqual(export_rows(iter(rows), output, "revision", limit=1), 1)
                self.assertEqual(export_rows(iter(rows), output, "revision"), 2)
                self.assertEqual(export_rows(iter(rows), output, "revision"), 2)
            self.assertEqual(writes, [(b"first", 16000, "FLOAT"), (b"third", 16000, "FLOAT")])
            manifest = [json.loads(l) for l in (output / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual([r["source_index"] for r in manifest], [0, 2])
            self.assertTrue(all(r["style"] == "original" for r in manifest))
            self.assertTrue(all(r["source_revision"] == "revision" for r in manifest))
            self.assertEqual(len(list((output / "audio").glob("*.wav"))), 2)

    def test_unexpected_diffa_answers_are_unresolved(self):
        self.assertEqual(diffa_prediction("YES!")["binary_prediction"], "neutral")
        self.assertEqual(diffa_prediction("no.")["binary_prediction"], "emotional")
        self.assertEqual(diffa_prediction("I'm not sure")["binary_prediction"], "unresolved")

    def test_external_classification_without_diffa_labels_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = [{"clip_id": str(i), "question_id": str(i), "style": "original",
                         "duration_seconds": 10, "audio_path": f"{i}.wav"} for i in range(2)]
            (path / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
            predict = Mock(side_effect=[{"emotion": "neutral"}, {"emotion": "happy"}])
            adapter = SimpleNamespace(MODEL="test", load_predictor=Mock(return_value=predict))
            fake_sf = SimpleNamespace(read=Mock(return_value=(SimpleNamespace(ndim=1), 16000)))
            with patch.object(sys, "argv", ["test", "--path", directory, "--model", "emotion2vec"]):
                with redirect_stdout(io.StringIO()), patch("classify_spoken_gsm.importlib.import_module",
                                                          return_value=adapter):
                    with patch.dict(sys.modules, {"soundfile": fake_sf, "librosa": SimpleNamespace()}):
                        main()
                    output = path / "results-emotion2vec-original.jsonl"
                    self.assertEqual(len(output.read_text().splitlines()), 2)
                    adapter.load_predictor.side_effect = AssertionError("Resume must not load model")
                    main()
            self.assertFalse((path / "results-diffa-original.jsonl").exists())
            report = json.loads((path / "classification-summary.json").read_text())
            self.assertEqual(report["models"]["emotion2vec"]["binary"], {"neutral": 1, "emotional": 1})
            self.assertEqual(report["agreement_with_diffa"], {})

    def test_agreement_is_only_computed_on_shared_valid_diffa_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            diffa = [{"clip_id": "a", **diffa_prediction("yes")},
                     {"clip_id": "b", **diffa_prediction("maybe")}]
            external = [{"clip_id": "a", "emotion": "neutral", "binary_prediction": "neutral"},
                        {"clip_id": "b", "emotion": "angry", "binary_prediction": "emotional"}]
            for name, rows in [("diffa", diffa), ("emotion2vec", external)]:
                (path / f"results-{name}-original.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows))
            with redirect_stdout(io.StringIO()):
                report = summarize(path, {"a": {}, "b": {}})
            self.assertEqual(report["agreement_with_diffa"]["emotion2vec"]["resolved_pairs"], 1)
            self.assertEqual(report["agreement_with_diffa"]["emotion2vec"]["agreement"], 1)


if __name__ == "__main__":
    unittest.main()
