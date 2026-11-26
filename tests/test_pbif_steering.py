from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import torch

import pbif_steering as run


class FullBufferSteeringTest(TestCase):
    def test_fit_and_apply_on_disjoint_requests(self):
        # With 64 answer slots, each half has its own value. Only averaging the
        # entire answer gives the expected vector. A very large prompt value
        # detects accidental inclusion of the text prefix.
        budget = 64
        blocks = [torch.nn.Linear(1, 1, bias=False) for _ in range(32)]
        block = blocks[31]
        class DecoderBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = block.weight

            def forward(self, hidden):
                return (hidden,)

        blocks[31] = DecoderBlock()
        original = {}
        for order in run.pbif.ORDERS:
            for style in run.pbif.STYLES:
                original["fit", f"{order}/{style}"] = dict(order=order, style=style)
        seen = []
        def generate(record):
            angry = record["style"] == "angry"
            calls = 3 if angry else 2
            for step in range(calls):
                hidden = torch.zeros(1, 1 + budget, 1)
                hidden[:, 0] = 1000
                if angry:
                    value = (4 if record["order"] == "tau+1" else 12) + step * 2
                    hidden[:, 1:33] = 900 if step == 2 else value
                result = blocks[31](hidden)[0]
                if record.get("question_id") == "held":
                    seen.append(result)
            return dict(response="test", response_tokens=1)

        tests = [dict(question_id="held", order=o, condition=f"{o}/{run.LABEL}", style="neutral")
                 for o in run.pbif.ORDERS]
        plan = dict(collection_ids=["fit"], test_ids=["held"], averaging="full", max_new_tokens=budget,
                    records=tests, recipe=dict(seed=42))
        with TemporaryDirectory() as tmp, mock.patch.object(run, "OUTPUT", Path(tmp)):
            folder = Path(tmp)
            run.freeze(folder/"run_config.json", plan)
            args = (plan, original, generate, blocks[31], folder/"fit.pt", folder/"vector.pt")
            run.fit_direction(*args)
            run.fit_direction(*args[:2], mock.Mock(side_effect=AssertionError), *args[3:])
            vec, info = run.read_vector(args[-1])
            self.assertEqual(info["extraction_ids"], ["fit"])
            self.assertEqual(vec.item(), 4.5)
            run.steer(plan, {}, generate, blocks, args[-1])
            self.assertEqual(len(seen), 4)
            for hidden in seen:
                self.assertEqual(hidden[0, 0, 0].item(), 1000)
                self.assertTrue((hidden[:, 1:] == 4.5).all())
            from run_diffa_spoken_prompt import saved
            rows = saved(folder, plan)
            self.assertEqual(set(rows), {("held", r["condition"]) for r in tests})
            run.steer(plan, rows, mock.Mock(side_effect=AssertionError), blocks, args[-1])
            self.assertEqual(len(blocks[31]._forward_hooks), 0)
