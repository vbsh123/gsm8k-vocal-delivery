"""Check held-out DIFFA-neutral selection without using math correctness."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_diffa_neutral_steering import main, select_remaining


class RemainingNeutralSteeringTests(unittest.TestCase):
    def test_prepares_frozen_cohort_and_invokes_existing_steer_with_fixed_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, previous, output = root / "data", root / "previous", root / "new"
            data.mkdir()
            previous.mkdir()
            records = [{"clip_id": str(i), "question_id": f"q{i}", "style": "neutral",
                        "speaker": "Ryan", "repeat": 0, "duration_seconds": 10} for i in range(5)]
            labels = [{"clip_id": str(i), "style": "neutral", "predicted_style": "yes"} for i in range(5)]
            def write(path, rows):
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            write(data / "manifest.jsonl", records)
            write(data / "results-neutral-binary.jsonl", labels)
            # No saved math answers: these should still be eligible.
            extraction, pilot, vectors = root / "extracted.jsonl", root / "pilot.jsonl", root / "vectors.pt"
            write(extraction, [{"question_id": "q0"}])
            write(pilot, [{"question_id": "q4"}])
            vectors.touch()
            (previous / "run_config.json").write_text(json.dumps({"recordings": [records[1]]}))
            argv = ["test", "--path", str(data), "--vectors", str(vectors),
                    "--previous-run", str(previous), "--extraction-questions", str(extraction),
                    "--pilot-results", str(pilot), "--output", str(output)]
            with patch.object(sys, "argv", argv + ["--prepare-only"]), redirect_stdout(io.StringIO()):
                with patch("run_diffa_neutral_steering.subprocess.run") as execute:
                    main()
                    execute.assert_not_called()
            selection = json.loads((output / "selection.json").read_text())
            self.assertEqual([r["question_id"] for r in selection["recordings"]], ["q2", "q3"])
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                with patch("run_diffa_neutral_steering.subprocess.run") as execute:
                    main()
                    command = execute.call_args.args[0]
            self.assertEqual(command[command.index("--num-examples") + 1], "2")
            self.assertEqual(command[command.index("--layers") + 1], "4")
            self.assertEqual(command[command.index("--groups") + 1], "acoustic")
            self.assertEqual(command[command.index("--strengths") + 1:command.index("--strengths") + 3], ["0", "1"])

    def test_excludes_training_and_previous_tests_but_includes_unevaluated(self):
        records = [{"clip_id": str(i), "question_id": f"q{i}", "style": "neutral",
                    "speaker": "Ryan", "repeat": 0, "duration_seconds": 10} for i in range(7)]
        labels = [{"clip_id": str(i), "style": "neutral", "predicted_style": "yes"}
                  for i in range(6)] + [{"clip_id": "6", "style": "neutral", "predicted_style": "no"}]
        answers = [{"clip_id": str(i), "correct": i % 2 == 0} for i in range(7) if i != 3]
        records[4]["duration_seconds"] = 31
        records[5]["repeat"] = 1
        selected, counts = select_remaining(records, labels, answers, {"q0"}, {"q1"})
        self.assertEqual([r["question_id"] for r in selected], ["q2", "q3"])
        self.assertEqual(counts["remaining"], 2)
        self.assertEqual(counts["with_saved_math_answer"], 3)
        self.assertEqual(counts["already_tested"], 1)
        self.assertEqual(counts["used_for_vector_extraction"], 1)
        for r in answers:
            r["correct"] = not r["correct"]
        selected2, _ = select_remaining(records, labels, answers, {"q0"}, {"q1"})
        self.assertEqual(selected2, selected)


if __name__ == "__main__":
    unittest.main()
