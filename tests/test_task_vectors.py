"""Outcome selection and real-autograd checks for the new steering methods."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from extract_benefit_vectors import angry_helped, main as extract_benefit
import evaluate_audio_steering
from train_answer_vectors import masked_answer_loss, trainable_hook

try:
    import torch
except ImportError:
    torch = None


class BenefitTests(unittest.TestCase):
    def test_selects_only_angry_correct_neutral_incorrect(self):
        manifest = {"neutral": {"answer": "18"}}
        for neutral, angry, expected in [("17", "18", True), ("18", "18", False),
                                         ("18", "17", False), ("17", "17", False)]:
            row = dict(pair=dict(neutral_id="neutral"), neutral_response="Final answer: " + neutral,
                       angry_response="Final answer: " + angry)
            self.assertEqual(angry_helped(row, manifest), expected)

    def test_both_methods_use_the_same_three_targets_and_questions(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = root/"source"
            source.mkdir()
            records = [dict(clip_id="clip", question_id="heldout", answer="18")]
            (source/"run_config.json").write_text(json.dumps(dict(recordings=records)))
            for method in ["benefit_mean", "answer_trained"]:
                folder = root/"vectors"/method
                folder.mkdir(parents=True)
                (folder/"style-vectors.pt").write_bytes(b"fixture")
                (folder/"extraction-questions.jsonl").write_text(json.dumps(dict(question_id="training"))+"\n")
            args = ["test", "--source-run", str(source), "--vectors-root", str(root/"vectors"),
                    "--methods", "benefit_mean", "answer_trained", "--include-l11",
                    "--num-examples", "1", "--output", str(root/"results"), "--prepare-only"]
            with patch.object(sys, "argv", args), patch.object(evaluate_audio_steering.subprocess, "run") as execute:
                evaluate_audio_steering.main()
            self.assertEqual(execute.call_count, 2)
            for method in ["benefit_mean", "answer_trained"]:
                config = json.loads((root/"results"/method/"run_config.json").read_text())
                self.assertEqual(config["recordings"], records)
                self.assertEqual(config["settings"]["seed"], 42)
                self.assertEqual([c["name"] for c in config["settings"]["conditions"]],
                                 ["baseline", "L11-answer-a1", "L12-acoustic-a1", "L16-answer-a1.25"])


@unittest.skipIf(torch is None, "PyTorch is needed for the gradient check")
class AnswerGradientTests(unittest.TestCase):
    def test_benefit_checkpoint_averages_only_wins_and_excludes_all_training_questions(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            collection = root/"collection"
            (collection/"pairs").mkdir(parents=True)
            pairs = [dict(question_id=f"q{i}", neutral_id=f"n{i}", angry_id=f"a{i}", seed=42) for i in range(2)]
            manifest = {f"n{i}": dict(question_id=f"q{i}", answer="18") for i in range(2)}
            plan = dict(pairs=pairs, manifest=manifest, prompt="fixed", max_new_tokens=512, averaging="all")
            (collection/"collection-config.json").write_text(json.dumps(plan))
            for i, pair in enumerate(pairs):
                torch.save(dict(pair=pair, difference=torch.full((2, 4, 3), 1.0 if i == 0 else 1000.0),
                                neutral_response="Final answer: " + ("17" if i == 0 else "18"),
                                angry_response="Final answer: 18"), collection/"pairs"/f"{i:05d}.pt")
            with patch.object(sys, "argv", ["test", "--collection", str(collection), "--output", str(root/"out")]):
                extract_benefit()
            state = torch.load(root/"out/benefit_mean/style-vectors.pt", weights_only=True)
            self.assertTrue(torch.equal(state["pair_mean_sums"]["angry"], torch.ones(2, 4, 3)))
            self.assertEqual(state["pair_counts"]["angry"].tolist(), [1, 1])
            self.assertEqual(state["extraction_question_ids"], ["q0", "q1"])
            self.assertEqual(state["averaging"], "all")

    def test_loss_reaches_only_the_vector_with_checkpointing_and_frozen_weights(self):
        from torch.utils.checkpoint import checkpoint

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, inputs):
                return (self.linear(inputs), None)

        torch.manual_seed(42)
        block = Block().eval().requires_grad_(False)
        head = torch.nn.Linear(4, 8).eval().requires_grad_(False)
        vector = torch.nn.Parameter(torch.zeros(4))
        inputs = torch.randn(1, 516, 4)
        targets = torch.zeros(512, dtype=torch.long)
        masked = torch.zeros(512, dtype=torch.bool)
        masked[:3] = True
        original = [p.detach().clone() for p in list(block.parameters()) + list(head.parameters())]
        optimizer = torch.optim.SGD([vector], lr=1)
        losses = []
        for _ in range(3):
            handle = block.register_forward_hook(trainable_hook(vector, slice(0, 2), 1.25))
            optimizer.zero_grad()
            try:
                hidden = checkpoint(block, inputs, use_reentrant=False)[0]
                logits = head(hidden + hidden.mean(dim=1, keepdim=True))
                loss = masked_answer_loss(logits, targets, masked, torch)
                loss.backward()
            finally:
                handle.remove()
            self.assertIsNotNone(vector.grad)
            self.assertTrue(torch.isfinite(vector.grad).all())
            self.assertGreater(float(vector.grad.norm()), 0)
            losses.append(float(loss.detach()))
            optimizer.step()
        self.assertLess(losses[-1], losses[0])
        for before, parameter in zip(original, list(block.parameters()) + list(head.parameters())):
            self.assertTrue(torch.equal(before, parameter))
            self.assertIsNone(parameter.grad)
        self.assertFalse(block._forward_hooks)


if __name__ == "__main__":
    unittest.main()
