"""Check all-style external-label accuracy and layer effects without model inference."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from analyze_layer_groups import analyze_run
from classify_evaluated import accuracy_rows, main


class ExternalGroupReportTests(unittest.TestCase):
    def test_accuracy_includes_all_styles_and_preserves_unknown(self):
        predictions = {
            "a": {"style": "neutral", "emotion": "neutral", "binary_prediction": "neutral"},
            "b": {"style": "angry", "emotion": "neutral", "binary_prediction": "neutral"},
            "c": {"style": "happy", "emotion": "<unk>", "binary_prediction": "unresolved"},
        }
        answers = {"a": {"correct": False, "response": "12.", "reference_answer": "12"},
                   "b": {"correct": False, "response": "Final answer: 3", "reference_answer": "4"},
                   "c": {"correct": True, "response": "Final answer: 5", "reference_answer": "5"}}
        rows = accuracy_rows(predictions, answers)
        pooled = next(r for r in rows if (r["generated_style"], r["group_type"], r["label"])
                      == ("all", "binary", "neutral"))
        self.assertEqual(pooled["count"], 2)
        self.assertEqual(pooled["strict"]["correct"], 0)
        self.assertEqual(pooled["non_strict"]["correct"], 1)
        self.assertTrue(any(r["generated_style"] == "happy" and r["label"] == "<unk>" for r in rows))

    def test_layer_effects_use_latest_condition_and_its_own_matched_baseline(self):
        def row(clip, condition, correct):
            return {"clip_id": clip, "condition": condition, "correct": correct,
                    "response": "Final answer: " + ("1" if correct else "2"), "reference_answer": "1"}
        rows = [row("a", "baseline", False), row("b", "baseline", True),
                row("a", "L4-acoustic-a1", False), row("a", "L4-acoustic-a1", True),
                row("b", "L4-acoustic-a1", False), row("c", "L4-acoustic-a1", True)]
        sources = {"emotion2vec": {"a": {"binary_prediction": "neutral", "emotion": "neutral"},
                                   "b": {"binary_prediction": "emotional", "emotion": "happy"}}}
        report = analyze_run(rows, sources, "non-strict", "run")
        neutral = next(r for r in report if r["group_type"] == "binary" and r["group"] == "neutral")
        happy = next(r for r in report if r["group_type"] == "emotion" and r["group"] == "happy")
        self.assertEqual((neutral["n"], neutral["wins"], neutral["losses"]), (1, 1, 0))
        self.assertEqual((happy["n"], happy["wins"], happy["losses"]), (1, 0, 1))

    def test_evaluated_runner_reuses_previous_labels_and_only_infers_missing_evaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            manifest = [{"clip_id": str(i), "question_id": str(i), "style": style,
                         "audio_path": f"{i}.wav"}
                        for i, style in enumerate(["neutral", "angry", "happy", "sad"])]
            answers = [{"clip_id": str(i), "correct": True, "response": "Final answer: 1",
                        "reference_answer": "1"} for i in range(3)]
            previous = [{"clip_id": "0", "question_id": "0", "style": "neutral", "model": "test",
                         "emotion": "neutral", "binary_prediction": "neutral"}]
            for name, rows in [("manifest.jsonl", manifest), ("results-reasoning-v2.jsonl", answers),
                               ("results-emotion2vec-neutral.jsonl", previous)]:
                (folder / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
            predict = Mock(return_value={"emotion": "happy"})
            adapter = SimpleNamespace(MODEL="test", load_predictor=Mock(return_value=predict))
            fake_sf = SimpleNamespace(read=Mock(return_value=(SimpleNamespace(ndim=1), 16000)))
            with patch.object(sys, "argv", ["test", "--model", "emotion2vec", "--path", directory]):
                with redirect_stdout(io.StringIO()), patch("classify_evaluated.importlib.import_module",
                                                          return_value=adapter):
                    with patch.dict(sys.modules, {"soundfile": fake_sf, "librosa": SimpleNamespace()}):
                        main()
                    self.assertEqual(predict.call_count, 2)
                    output = folder / "results-emotion2vec-evaluated.jsonl"
                    saved = [json.loads(line) for line in output.read_text().splitlines()]
                    self.assertEqual([r["clip_id"] for r in saved], ["0", "1", "2"])
                    self.assertEqual([r["style"] for r in saved], ["neutral", "angry", "happy"])
                    adapter.load_predictor.side_effect = AssertionError("Resume must not load model")
                    main()
                    self.assertEqual(len(output.read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
