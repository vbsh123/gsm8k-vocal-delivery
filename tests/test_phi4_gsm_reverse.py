from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import steer_phi4_gsm_reverse as run


class PhiReverseTests(unittest.TestCase):
    def test_native_generation_kwargs_suffix_and_observer_parity(self):
        class Batch(dict):
            def to(self, *args, **kwargs): return self
        class Processor:
            def __call__(self, **kwargs):
                self.inputs = kwargs
                return Batch(input_ids=torch.tensor([[7, 8]]))
            def batch_decode(self, ids, **kwargs):
                self.decoded = ids.clone(); self.options = kwargs
                return ['#### 2']
        class Block(torch.nn.Module):
            def forward(self, h): return h, None
        native = run.NativePhi.__new__(run.NativePhi)
        native.processor = Processor(); native.layer = Block(); native.config = object()
        recorded = []
        def generate(**kwargs):
            recorded.append(kwargs)
            native.layer(torch.tensor([[[100., 100.], [1., 3.]]]))
            native.layer(torch.tensor([[[3., 5.]]]))
            return torch.tensor([[7, 8, 20, 21]])
        native.model = SimpleNamespace(device='cpu', generate=generate)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'audio'; path.write_bytes(b'original audio')
            row = dict(audio_file=str(path), audio_sha256=run.digest(path))
            with patch.object(run, 'load_audio', return_value=[0.]*160):
                baseline = native.generate(row, 'Original prompt')
                observed = native.generate(row, 'Original prompt', observe=True)
        self.assertEqual(baseline['response'], observed['response'])
        self.assertEqual(observed['response_tokens'], 2)
        self.assertEqual(observed['hook_forwards'], 2)
        torch.testing.assert_close(observed['mean'], torch.tensor([2., 4.]))
        torch.testing.assert_close(native.processor.decoded, torch.tensor([[20, 21]]))
        for kwargs in recorded:
            for key, value in dict(max_new_tokens=512, do_sample=False, num_beams=1, repetition_penalty=1., use_cache=True).items():
                self.assertEqual(kwargs[key], value)
        self.assertFalse(native.layer._forward_hooks)

    def fixture(self, root):
        source = root/'audio'; source.mkdir(); output = root/'out'; output.mkdir()
        questions = [dict(question_id=f'q{i:04d}', question='Question', answer='2') for i in range(800)]
        rows = [dict(question_id=q['question_id'], voice='Jenny', style=s, condition=s,
                     audio_file=f'/audio/{q["question_id"]}-{s}.wav', audio_sha256=q['question_id']+s,
                     seconds=45., reference_answer='2') for q in questions for s in ('neutral', 'angry')]
        parent = dict(recipe=dict(run.MODELS['phi4']), recordings=rows)
        args = SimpleNamespace(source=source, output=output)
        with patch.object(run.gsm, 'make_plan', return_value=parent), patch.object(run.gsm, 'questions', return_value=questions):
            plan = run.prepare(args)
        return args, plan

    def test_frozen400_400_no_selection_by_answers_and_no_test_angry(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            args, plan = self.fixture(Path(d))
            self.assertEqual(len(plan['collection']), 400)
            self.assertEqual(len(plan['test']), 400)
            train = {p['question_id'] for p in plan['collection']}
            test = {p['question_id'] for p in plan['test']}
            self.assertFalse(train & test)
            self.assertTrue(all(r['style'] == 'neutral' and r['seconds'] == 45 for r in plan['test']))
            self.assertEqual(plan['prompt'], run.gsm.ORIGINAL)
            self.assertEqual(list(plan['strengths'].values()), [0, -.5, -1])
            self.assertEqual(plan, run.prepare(args))

    def test_hook_only_last_position_sign_tuple_cache_and_cleanup(self):
        class Block(torch.nn.Module):
            def forward(self, h): return h, 'unchanged cache'
        block = Block(); h = torch.arange(12.).reshape(1, 4, 3); v = torch.tensor([2., 4., 6.])
        with run.answer_hook(block, v, -.5) as state:
            actual, cache = block(h)
            torch.testing.assert_close(actual[:, :-1], h[:, :-1])
            torch.testing.assert_close(actual[:, -1], h[:, -1]-.5*v)
            self.assertEqual(cache, 'unchanged cache')
            cached, _ = block(h[:, -1:])
            torch.testing.assert_close(cached, h[:, -1:]-v*.5)
        self.assertEqual(state['count'], 2)
        self.assertFalse(block._forward_hooks)
        with run.answer_hook(block, v, 0) as zero:
            self.assertFalse(block._forward_hooks)
            torch.testing.assert_close(block(h)[0], h)
        self.assertEqual(zero['count'], 0)
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            with run.answer_hook(block, v, -1): raise RuntimeError('native failure')
        self.assertFalse(block._forward_hooks)

    def test_observer_averages_prediction_positions_not_whole_prompt(self):
        class Block(torch.nn.Module):
            def forward(self, h): return (h,)
        block = Block()
        h = torch.tensor([[[100., 100.], [1., 3.]]])
        with run.answer_hook(block, observe=True) as state:
            torch.testing.assert_close(block(h)[0], h)
            block(torch.tensor([[[3., 5.]]]))
        torch.testing.assert_close(state['total']/state['count'], torch.tensor([2., 4.]))

    def test_collect_equal_question_weight_resume_sign_and_evaluate_only_test(self):
        class Native:
            def __init__(self): self.calls = []
            def generate(self, row, prompt, direction=None, strength=0, observe=False):
                self.calls.append((row['question_id'], row['style'], strength, observe))
                result = dict(response='#### 2', response_tokens=4, hook_forwards=1)
                if observe:
                    index = int(row['question_id'][1:])
                    result.update(mean=torch.full((3072,), float(index+1) if row['style'] == 'angry' else 0.),
                                  hook_forwards=1+index%5)
                else:
                    if strength:
                        assert direction.shape == (3072,)
                        assert row['style'] == 'neutral'
                return result
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            args, plan = self.fixture(Path(d)); native = Native()
            receipt = run.digest(args.output/'run_config.json')
            vector = run.collect(plan, args.output, native, receipt)
            self.assertEqual(len(native.calls), 800)
            torch.testing.assert_close(vector, torch.full((3072,), 200.5))
            torch.testing.assert_close(run.collect(plan, args.output, native, receipt), vector)
            self.assertEqual(len(native.calls), 800)  # No collection rerun.
            run.evaluate(plan, args.output, native, vector, receipt)
            self.assertEqual(len(native.calls), 2000)
            rows, done = run.completed(plan, args.output, receipt)
            self.assertEqual(len(rows), 1200)
            self.assertTrue(all(r['question_id'] in plan['bank'] and r['style'] == 'neutral' for r in rows))
            self.assertEqual({r['strength'] for r in rows}, {0., -.5, -1.})
            run.evaluate(plan, args.output, native, vector, receipt)
            self.assertEqual(len(native.calls), 2000)  # No test rerun.
            corrupted = dict(rows[0], strength=-2)
            (args.output/'results.jsonl').write_text(json.dumps(corrupted)+'\n')
            with self.assertRaisesRegex(ValueError, 'protocol'): run.completed(plan, args.output, receipt)


if __name__ == '__main__': unittest.main()
