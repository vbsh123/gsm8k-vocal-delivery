"""Check neutral-only selection, binary mapping, agreement denominators and resume."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from emotion_agreement import agreement_summary, binary_label, run
from emotion2vec_agreement import parse_prediction as emotion2vec_prediction
from sensevoice_agreement import parse_prediction as sensevoice_prediction
from speechbrain_agreement import parse_prediction as speechbrain_prediction


class AgreementTests(unittest.TestCase):
    def test_speechbrain_label_encoder_order_and_scores(self):
        # Use the checkpoint's encoder order, not a presumed alphabetical order.
        labels = ["hap", "ang", "neu", "sad"]
        prediction = speechbrain_prediction(labels, [0.1, 0.05, 0.8, 0.05])
        self.assertEqual(prediction["emotion"], "neutral")
        self.assertEqual(prediction["predicted_label"], "neu")
        self.assertEqual(prediction["scores"][2], 0.8)
        self.assertAlmostEqual(sum(prediction["scores"]), 1)
        prediction = speechbrain_prediction(labels, [0.1, 0.7, 0.1, 0.1])
        self.assertEqual(binary_label(prediction["emotion"]), "emotional")

    def test_actual_model_output_formats(self):
        prediction = emotion2vec_prediction({"labels": ["中立/neutral", "生气/angry"],
                                            "scores": [0.2, 0.8]})
        self.assertEqual(prediction["emotion"], "angry")
        prediction = sensevoice_prediction({"text": "<|en|><|NEUTRAL|><|Speech|><|woitn|>Hello"})
        self.assertEqual(prediction["emotion"], "neutral")
        self.assertEqual(sensevoice_prediction({"text": "<|en|><|EMO_UNKNOWN|>Hi"})["emotion"],
                         "emo_unknown")
        self.assertEqual(sensevoice_prediction({"text": "No emotion token"})["emotion"], "unknown")
        for label in ("unknown", "other", "emo_unknown"):
            self.assertEqual(binary_label(label), "unresolved")

    def test_group_agreement_kappa_and_unknown_coverage(self):
        labels = {str(i): {"predicted_style": label}
                  for i, label in enumerate(["yes", "yes", "no", "no", "yes"])}
        predictions = {str(i): {"binary_prediction": prediction} for i, prediction in enumerate(
            ["neutral", "emotional", "emotional", "emotional", "unresolved"])}
        summary = agreement_summary(dict.fromkeys(labels), labels, predictions)
        self.assertEqual(summary["groups"]["neutral"]["selected"], 3)
        self.assertEqual(summary["groups"]["neutral"]["agreement"], 0.5)
        self.assertEqual(summary["groups"]["neutral"]["unresolved"], 1)
        self.assertEqual(summary["groups"]["emotional"]["agreement"], 1)
        self.assertEqual(summary["resolved_pairs"], 4)
        self.assertEqual(summary["agreement"], 0.75)
        self.assertEqual(summary["cohen_kappa"], 0.5)

    def test_runner_only_selects_classified_neutrals_and_resumes_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            recordings = [{"clip_id": str(i), "question_id": str(i), "style": style,
                           "audio_path": f"{i}.wav"}
                          for i, style in enumerate(["neutral", "neutral", "angry", "neutral"])]
            labels = [{"clip_id": str(i), "predicted_style": label}
                      for i, label in enumerate(["yes", "no", "no", "invalid"])]
            for name, rows in [("manifest.jsonl", recordings), ("results-neutral-binary.jsonl", labels)]:
                (folder / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
            predictor = Mock(side_effect=[{"emotion": "neutral"}, {"emotion": "happy"}])
            loader = Mock(return_value=predictor)
            audio = SimpleNamespace(ndim=1)
            fake_sf = SimpleNamespace(read=Mock(return_value=(audio, 16000)))
            with patch.object(sys, "argv", ["test", "--path", directory]), redirect_stdout(io.StringIO()):
                with patch.dict(sys.modules, {"soundfile": fake_sf, "librosa": SimpleNamespace()}):
                    run("test", "test-model", loader)
                output = folder / "results-test-neutral.jsonl"
                saved = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual([r["clip_id"] for r in saved], ["0", "1"])
                self.assertEqual(predictor.call_count, 2)
                no_load = Mock(side_effect=AssertionError("Resume must not load the model"))
                run("test", "test-model", no_load)
                self.assertEqual(len(output.read_text().splitlines()), 2)
                summary = json.loads(output.with_suffix(".summary.json").read_text())
                self.assertEqual(summary["agreement"], 1)
            # The filtered-angry comparison selects DIFFA-emotional angry clips,
            # writes separately, and leaves all neutral predictions intact.
            recordings.append({"clip_id": "4", "question_id": "4", "style": "angry",
                               "audio_path": "4.wav"})
            labels.append({"clip_id": "4", "predicted_style": "yes"})
            for name, rows in [("manifest.jsonl", recordings), ("results-neutral-binary.jsonl", labels)]:
                (folder / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
            angry_loader = Mock(return_value=Mock(return_value={"emotion": "angry"}))
            with patch.object(sys, "argv", ["test", "--path", directory,
                                           "--style", "angry", "--diffa-label", "no"]):
                with redirect_stdout(io.StringIO()), patch.dict(
                    sys.modules, {"soundfile": fake_sf, "librosa": SimpleNamespace()}
                ):
                    run("test", "test-model", angry_loader)
            angry_output = folder / "results-test-angry.jsonl"
            angry_saved = [json.loads(line) for line in angry_output.read_text().splitlines()]
            self.assertEqual([r["clip_id"] for r in angry_saved], ["2"])
            self.assertEqual(angry_saved[0]["style"], "angry")
            self.assertEqual(angry_saved[0]["diffa_prediction"], "no")
            self.assertEqual(len(output.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
