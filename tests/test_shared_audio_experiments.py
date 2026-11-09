"""Check full-trajectory averaging and the shared seed42 steering loop without a GPU."""

from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from extract_audio_vectors import average_pair
from steer_audio import main, run_experiment
from test_steering import Hidden, Vector
from test_steering_followup import Layer


class Values(list):
    def __sub__(self, other):
        return Values(a-b for a, b in zip(self, other))


class Trajectory:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key):
        return Trajectory([row[key[1]] for row in self.rows[key[0]]])

    def __sub__(self, other):
        return Trajectory([[a-b for a, b in zip(row, other_row)]
                           for row, other_row in zip(self.rows, other.rows)])

    def mean(self, dim):
        assert dim == 1
        return Values(sum(row)/len(row) for row in self.rows)


class Direction(Vector):
    def __truediv__(self, count):
        return Direction(v/count for v in self)

    def to(self, **kwargs):
        return self


class SharedAudioTests(unittest.TestCase):
    def test_all_forwards_includes_tails_but_matched_drops_them_per_layer(self):
        neutral = Trajectory([[1, 3], [2, 4]])
        angry = Trajectory([[4, 6, 20, 30], [5, 7, 21, 31]])
        _, _, matched, steps = average_pair(neutral, angry, "matched")
        _, _, all_forwards, unused = average_pair(neutral, angry, "all")
        self.assertEqual((matched, steps), ([3, 3], 2))
        self.assertEqual((all_forwards, unused), ([13, 13], None))

    def test_runner_resets_seed_and_removes_hooks_between_conditions_and_on_resume(self):
        layers = [Layer(), Layer()]
        seeds, generations = [], []
        decoder = ModuleType("src.fast_dllm.diffa_generate")
        decoder.insert_audio_embedding = lambda *args: None
        original_insert = decoder.insert_audio_embedding

        class Features:
            input_features = "features"
            attention_mask = "mask"

            def to(self, device):
                return self

        def generate(*args, **kwargs):
            self.assertEqual(kwargs["input_texts"], ["fixed prompt"])
            self.assertEqual((kwargs["steps"], kwargs["block_length"], kwargs["max_new_tokens"]), (128, 32, 2))
            decoder.insert_audio_embedding(None, SimpleNamespace(shape=(1, 4)), [SimpleNamespace(item=lambda: 1)])
            hidden = Hidden([[0, 0] for _ in range(7)])
            for layer in layers:
                hidden = layer.run(hidden)
            generations.append(hidden.rows)
            return "Final answer: 1"

        model = SimpleNamespace(config=SimpleNamespace(prompt_size=2),
                                llm_model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=layers))),
                                generate=generate)
        checkpoint = dict(prompt="fixed prompt", max_new_tokens=2, groups=["answer"],
                          pair_counts={"angry": [1, 1]},
                          pair_mean_sums={"angry": {(0, 0): Direction([2, 2]), (1, 0): Direction([4, 4])}})
        torch = SimpleNamespace(load=lambda *a, **k: checkpoint, bfloat16="bf16",
                                manual_seed=seeds.append, cuda=SimpleNamespace(manual_seed_all=seeds.append),
                                inference_mode=nullcontext)
        modules = {
            "librosa": SimpleNamespace(),
            "numpy": SimpleNamespace(random=SimpleNamespace(seed=seeds.append)),
            "soundfile": SimpleNamespace(read=lambda *a, **k: (SimpleNamespace(ndim=1), 16000)),
            "torch": torch,
            "src": ModuleType("src"),
            "src.fast_dllm": ModuleType("src.fast_dllm"),
            "src.fast_dllm.diffa_generate": decoder,
            "extract_vectors": SimpleNamespace(load_model=lambda: (model, "tokenizer", lambda *a, **k: Features())),
        }
        conditions = [dict(name="baseline", interventions=[]),
                      dict(name="combo", interventions=[dict(layer=0, group="answer", strength=1),
                                                       dict(layer=1, group="answer", strength=0.5)])]
        records = [dict(clip_id="clip", question_id="question", audio_path="audio.wav", answer="1")]
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", modules), redirect_stdout(io.StringIO()):
            folder = Path(directory)
            run_experiment(records, conditions, folder/"vectors.pt", folder)
            rows = [json.loads(line) for line in (folder/"results.jsonl").read_text().splitlines()]
            self.assertEqual([r["seed"] for r in rows], [42, 42])
            self.assertEqual([r["steered_forwards"] for r in rows], [0, 1])
            self.assertEqual(generations[0][-2:], [[0, 0], [0, 0]])
            self.assertEqual(generations[1][-2:], [[4, 4], [4, 4]])
            self.assertEqual(seeds, [42]*6)
            self.assertTrue(all(not layer.hooks for layer in layers))
            self.assertIs(decoder.insert_audio_embedding, original_insert)
            run_experiment(records, conditions, folder/"vectors.pt", folder)
            self.assertEqual(len(generations), 2)

    def test_preparing_and_resuming_a_plan_keeps_the_same_questions_and_conditions(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            folder = Path(directory)
            plan = folder/"plan.json"
            vectors = folder/"vectors.pt"
            vectors.write_bytes(b"fixture")
            source = dict(recordings=[dict(clip_id="one", question_id="q")],
                          conditions=[dict(name="baseline", interventions=[])])
            plan.write_text(json.dumps(source))
            args = ["--plan", str(plan), "--vectors", str(vectors), "--output", str(folder/"run"), "--prepare-only"]
            main(args)
            before = (folder/"run/run_config.json").read_text()
            main(args)
            saved = json.loads(before)
            self.assertEqual(saved["recordings"], source["recordings"])
            self.assertEqual(saved["settings"]["conditions"], source["conditions"])
            self.assertEqual(saved["settings"]["seed"], 42)
            self.assertEqual((folder/"run/run_config.json").read_text(), before)


if __name__ == "__main__":
    unittest.main()
