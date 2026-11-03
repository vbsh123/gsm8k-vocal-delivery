"""Check original-audio classification precedes matched steering, with no regeneration."""

from contextlib import redirect_stdout, redirect_stderr
import ast
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_spoken_steering import main, validate_classifications
from classify_spoken_gsm import DIFFA_MODEL, DIFFA_PROMPT, diffa_prediction
from steer import reuse_questions


class SpokenSteeringTests(unittest.TestCase):
    def test_classifier_prompt_matches_existing_generated_audio_filter(self):
        source = Path(__file__).resolve().parents[1] / "filter.py"
        assignments = [node for node in ast.walk(ast.parse(source.read_text()))
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "prompt" for target in node.targets)]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(ast.literal_eval(assignments[0].value), DIFFA_PROMPT)

    def test_wrong_classification_model_or_prompt_is_rejected(self):
        original = {"model": DIFFA_MODEL, **diffa_prediction("yes")}
        validate_classifications({"a": original}, {"a": {}})
        for field, value in [("model", "wrong-model"), ("prompt", "different question"),
                             ("max_new_tokens", 32), ("decoder", "different-decoder")]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_classifications({"a": {**original, field: value}}, {"a": {}})

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
                    labels = [{"clip_id": str(i), "model": DIFFA_MODEL, **diffa_prediction(label)}
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
                    # Changed audio metadata must fail before any GPU subprocess.
                    changed = [dict(r) for r in records]
                    changed[0]["answer"] = "999"
                    (data / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in changed))
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main()
                    self.assertEqual(len(calls), 3)
                    (data / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
                    # A reclassified input cannot silently remain in the neutral cohort.
                    labels = [{"clip_id": str(i), "model": DIFFA_MODEL, **diffa_prediction("no")}
                              for i in range(3)]
                    (data / "results-diffa-original.jsonl").write_text(
                        "".join(json.dumps(r) + "\n" for r in labels))
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        main()
                    self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
