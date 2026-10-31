"""Check pair selection and first-pass capture without model/GPU dependencies."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from extract_vectors import capture_first_pass, make_pairs, question_split


class PairSelectionTests(unittest.TestCase):
    def row(self, q, style, seconds=10, speaker="Ryan"):
        clip = f"{q}-{speaker}-{style}"
        return dict(clip_id=clip, question_id=q, style=style, speaker=speaker,
                    repeat=0, duration_seconds=seconds)

    def test_short_pair_survives_an_unrelated_long_style(self):
        rows = [self.row("q1", "neutral"), self.row("q1", "angry"), self.row("q1", "happy", 31)]
        pairs = make_pairs({r["clip_id"]: r for r in rows}, ["angry", "happy"], question_split("q1", 42), 42, 0)
        self.assertEqual([p["style"] for p in pairs], ["angry"])

    def test_long_neutral_and_mismatched_speaker_do_not_pair(self):
        for neutral in [self.row("q1", "neutral", 31), self.row("q1", "neutral", speaker="Aiden")]:
            rows = [neutral, self.row("q1", "angry")]
            pairs = make_pairs({r["clip_id"]: r for r in rows}, ["angry"], question_split("q1", 42), 42, 0)
            self.assertEqual(pairs, [])

    def test_question_limit_keeps_styles_together_and_splits_do_not_overlap(self):
        rows = [self.row(f"q{i}", s) for i in range(100) for s in ["neutral", "angry", "happy"]]
        records = {r["clip_id"]: r for r in rows}
        splits = [make_pairs(records, ["angry", "happy"], s, 42, 0)
                  for s in ["train", "validation", "test"]]
        ids = [{p["question_id"] for p in pairs} for pairs in splits]
        self.assertEqual(len(set.union(*ids)), 100)
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        limited = make_pairs(records, ["angry", "happy"], "train", 42, 2)
        self.assertEqual(len(limited), 4)
        self.assertEqual(len({p["question_id"] for p in limited}), 2)


class Hidden:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        batch, positions = key
        assert batch == 0
        return Hidden(self.rows[positions])

    def float(self):
        return self

    def mean(self, dim):
        assert dim == 0
        values = [sum(column) / len(column) for column in zip(*self.rows)]
        return SimpleNamespace(cpu=lambda: values)


class Block:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def run(self, hidden):
        for hook in list(self.hooks):
            hook(self, (), (hidden, None))


class CaptureTests(unittest.TestCase):
    def test_answer_positions_only_single_pass_and_hooks_removed(self):
        blocks = [Block(), Block()]
        calls = []

        def generate(*args, **kwargs):
            calls.append(kwargs)
            blocks[0].run(Hidden([[999, 999], [2, 4], [4, 8]]))
            blocks[1].run(Hidden([[999, 999], [6, 8], [8, 12]]))
            self.fail("Extraction should stop before logits or subsequent passes")

        model = SimpleNamespace(llm_model=SimpleNamespace(model=SimpleNamespace(
            transformer={"blocks": blocks})), generate=generate)
        # Match PyTorch ModuleDict's attribute access.
        class Transformer(dict):
            def __getattr__(self, name):
                return self[name]
        model.llm_model.model.transformer = Transformer(blocks=blocks)
        fake_torch = SimpleNamespace(inference_mode=nullcontext, stack=lambda xs: xs)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            result = capture_first_pass(model, None, SimpleNamespace(input_features=1, attention_mask=2), "solve", 2)
        self.assertEqual(result, [[3, 6], [7, 10]])
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(not b.hooks for b in blocks))

    def test_hooks_removed_on_real_error(self):
        block = Block()
        class Transformer(dict):
            def __getattr__(self, name):
                return self[name]
        def generate(*args, **kwargs):
            raise ValueError("real failure")
        model = SimpleNamespace(llm_model=SimpleNamespace(model=SimpleNamespace(
            transformer=Transformer(blocks=[block]))), generate=generate)
        with patch.dict("sys.modules", {"torch": SimpleNamespace(inference_mode=nullcontext)}):
            with self.assertRaisesRegex(ValueError, "real failure"):
                capture_first_pass(model, None, SimpleNamespace(input_features=1, attention_mask=2), "solve", 2)
        self.assertEqual(block.hooks, [])
