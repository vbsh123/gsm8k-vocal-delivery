"""Check baseline compatibility, separate conditions, and no duplicate baseline copies."""

from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_spoken_layer_sweep import import_baselines, main, source_baselines
from scoring import score_response
import steer


def baseline(recording, seed=42):
    response = "Final answer: 1"
    return {"clip_id": recording["clip_id"], "question_id": recording["question_id"],
            "condition": "baseline", "seed": int.from_bytes(hashlib.sha256(
                f"{seed}:{recording['clip_id']}".encode()).digest()[:4], "big"),
            "steered_forwards": 0, "forwards": 2, "strength": 0,
            "prompt": "Solve the audio. Final answer: <number>", "max_new_tokens": 512,
            "response": response, **score_response(response, recording["answer"])}


class SpokenLayerSweepTests(unittest.TestCase):
    def test_full_prepare_reuses_baselines_and_builds_22_separate_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, source, output = root / "data", root / "source", root / "new"
            data.mkdir()
            source.mkdir()
            vectors, exclusions = root / "vectors.pt", source / "excluded.jsonl"
            vectors.write_bytes(b"fixture")
            exclusions.write_text("")
            records = [{"clip_id": str(i), "question_id": f"spoken-{i}", "style": "original",
                        "duration_seconds": 10, "audio_path": f"{i}.wav", "answer": str(i+1)} for i in range(2)]
            (data / "manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
            stat = vectors.stat()
            settings = dict(path=str(data), vectors=str(vectors), exclude_results=[str(exclusions)],
                            seed=42, style="angry", mode="fixed", min_pairs=1,
                            vector_bytes=stat.st_size, vector_mtime_ns=stat.st_mtime_ns)
            (source / "run_config.json").write_text(json.dumps({"settings": settings, "recordings": records}))
            baselines = [baseline(r) for r in records]
            (source / "results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in baselines))
            calls = []
            def execute(command, **kwargs):
                calls.append(command)
                self.assertIn("--prepare-only", command)
                filename = next(v for v in command if v.endswith("steer.py"))
                with patch.object(sys, "argv", [filename, *command[command.index(filename)+1:]]):
                    steer.main()
                return __import__("subprocess").CompletedProcess(command, 0, "", "")
            with patch.object(sys, "argv", ["test", "--source-run", str(source), "--output", str(output),
                                           "--prepare-only"]), redirect_stdout(io.StringIO()):
                with patch("run_spoken_layer_sweep.subprocess.run", side_effect=execute):
                    main()
                    main()
            saved = [json.loads(l) for l in (output / "results.jsonl").read_text().splitlines()]
            self.assertEqual(saved, baselines)
            config = json.loads((output / "run_config.json").read_text())
            conditions = config["settings"]["conditions"]
            self.assertEqual(len(conditions), 23)
            self.assertEqual({c["layer"] for c in conditions[1:]}, set(range(15, 26)))
            self.assertEqual({c["group"] for c in conditions[1:]}, {"acoustic", "answer"})
            self.assertTrue(all(c["strength"] == 1 for c in conditions[1:]))
            self.assertEqual(config["recordings"], records)
            self.assertEqual(len(calls), 2)

    def test_baseline_seed_scoring_and_missing_rows_are_checked(self):
        recording = {"clip_id": "a", "question_id": "qa", "answer": "1"}
        plan = {"settings": {"seed": 42}, "recordings": [recording]}
        row = baseline(recording)
        self.assertEqual(source_baselines(plan, [row]), [row])
        for change in ({"seed": 1}, {"steered_forwards": 1}, {"correct": False}, {"reference_answer": "2"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                source_baselines(plan, [{**row, **change}])
        with self.assertRaises(ValueError):
            source_baselines(plan, [])

    def test_different_existing_baseline_is_not_overwritten(self):
        recording = {"clip_id": "a", "question_id": "qa", "answer": "1"}
        row = baseline(recording)
        conditions = steer.make_conditions([15], ["acoustic"], [0, 1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(json.dumps({**row, "response": "different"})+"\n")
            before = path.read_text()
            with self.assertRaises(ValueError):
                import_baselines(path, [row], [recording], conditions)
            self.assertEqual(path.read_text(), before)


if __name__ == "__main__":
    unittest.main()
