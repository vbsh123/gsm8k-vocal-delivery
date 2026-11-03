"""Check original-audio classification precedes matched steering, with no regeneration."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_spoken_steering import main
from steer import reuse_questions


class SpokenSteeringTests(unittest.TestCase):
    def test_classify_then_steer_only_originals_predicted_neutral_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, output = root / "data", root / "steering"
            data.mkdir()
            vectors = root / "vectors.pt"
            vectors.touch()
            records = [{"clip_id": str(i), "question_id": f"spoken_{i}", "style": "original",
                        "speaker": "source", "repeat": 0, "duration_seconds": 12,
                        "audio_path": f"{i}.wav", "answer": "1"} for i in range(3)]
            (data / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            calls = []
            def execute(command, **kwargs):
                calls.append(command)
                if any(Path(value).name == "classify_spoken_gsm.py" for value in command):
                    labels = [{"clip_id": str(i), "predicted_style": label}
                              for i, label in enumerate(["yes", "no", "yes"])]
                    (data / "results-diffa-original.jsonl").write_text(
                        "".join(json.dumps(r) + "\n" for r in labels))
            argv = ["test", "--path", str(data), "--vectors", str(vectors),
                    "--output", str(output), "--num-examples", "3"]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                with patch("run_spoken_steering.subprocess.run", side_effect=execute):
                    main()
                    self.assertEqual(len(calls), 2)
                    selection = json.loads((output / "selection.json").read_text())["recordings"]
                    self.assertEqual([r["clip_id"] for r in selection], ["0", "2"])
                    self.assertEqual(reuse_questions(selection, {r["clip_id"]: r for r in records}, set(), 2), selection)
                    self.assertIn("--layers", calls[-1])
                    self.assertEqual(calls[-1][calls[-1].index("--groups")+1], "acoustic")
                    self.assertEqual(calls[-1][calls[-1].index("--num-examples")+1], "2")
                    main()
                    # Frozen selection bypasses reclassification on restart.
                    self.assertEqual(len(calls), 3)
                    self.assertTrue(any(Path(value).name == "steer.py" for value in calls[-1]))


if __name__ == "__main__":
    unittest.main()
