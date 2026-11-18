from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import steer_minicpm_gsm as run


class MiniSteeringTests(unittest.TestCase):
    def test_native_answer_hook_prefill_cached_steps_and_cleanup(self):
        class Block(torch.nn.Module):
            def forward(self, h): return h, 'cache-preserved'
        layer = Block(); seen = []
        def chat(**kwargs):
            for h in (torch.zeros(1, 3, 4096), torch.zeros(1, 1, 4096)):
                out = layer(h); seen.append(out[0].clone())
                self.assertEqual(out[1], 'cache-preserved')
            return '#### 2'
        native = run.NativeMini.__new__(run.NativeMini)
        native.layer = layer; native.model = SimpleNamespace(chat=chat)
        native.options = {'do_sample': False}
        native.processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **k: [1, 2]))
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d)/'audio'; audio.write_bytes(b'audio')
            row = dict(audio_file=str(audio), audio_sha256=run.digest(audio))
            with patch.object(run.diagnostics, 'load_audio', return_value=[]):
                result = native.generate(row, 'Prompt', torch.ones(4096), 1.)
                baseline = native.generate(row, 'Prompt', observe=True)
        self.assertEqual(result['response_tokens'], 2)
        self.assertTrue(torch.equal(seen[0][0, :-1], torch.zeros(2, 4096)))
        self.assertTrue(torch.equal(seen[0][0, -1], torch.ones(4096)))
        self.assertTrue(torch.equal(seen[1], torch.ones(1, 1, 4096)))
        self.assertEqual(baseline['hook_forwards'], 2)
        torch.testing.assert_close(baseline['mean'], torch.zeros(4096))
        self.assertFalse(layer._forward_hooks)

    def test_frozen_split_collection_vector_and_resume(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); output = root/'out'; output.mkdir(); diagnostic = root/'diagnostic'; diagnostic.mkdir()
            source = root/'source'; source.mkdir(); bank_path = root/'bank'; bank_path.write_text('bank')
            questions = [dict(question_id=f'q{i}', question='Q', answer='2') for i in range(1269)]
            records = [dict(question_id=q['question_id'], voice='Jenny', style=s, condition=s,
                            reference_answer='2', audio_file='audio', audio_sha256=q['question_id']+s)
                       for q in questions for s in ('neutral', 'angry')]
            prior = dict(protocol=run.diagnostics.PROTOCOL, recipe=run.diagnostics.SPEC,
                         reasoning_prompt='Prompt', native_options={'do_sample': False})
            (diagnostic/'run_config.json').write_text(json.dumps(prior))
            args = SimpleNamespace(output=output, source=source, diagnostics=diagnostic, gpu_budget_gib=22.)
            parent = dict(recordings=records, prompt='Prompt', recipe=dict(run.diagnostics.SPEC))
            with patch.object(run.gsm, 'make_plan', return_value=parent), patch.object(run.gsm, 'questions', return_value=questions), patch.object(run.gsm, 'QUESTION_FILE', bank_path):
                plan = run.prepare(args)
                self.assertEqual(plan, run.prepare(args))
            self.assertEqual(len(plan['collection']), 400); self.assertEqual(len(plan['test']), 869)
            self.assertFalse({p['question_id'] for p in plan['collection']}&set(plan['bank']))
            self.assertEqual(plan['strengths'], {'neutral': 0., 'neutral-L30-answer-a1': 1.})
            tiny = dict(plan, collection=plan['collection'][:2]); calls = []
            def generate(row, prompt, observe=False):
                calls.append(row)
                return dict(mean=torch.full((4096,), 3. if row['style']=='angry' else 1.),
                            hook_forwards=2, response='#### 2', response_tokens=2)
            native = SimpleNamespace(generate=generate)
            vector = run.collect(tiny, output, native, 'config')
            torch.testing.assert_close(vector, torch.full((4096,), 2.))
            run.collect(tiny, output, native, 'config'); self.assertEqual(len(calls), 4)
            cache = torch.load(output/'pairs/0000.pt', weights_only=True)
            cache['answer_difference'] *= -1
            with self.assertRaisesRegex(ValueError, 'sign/value'):
                run.validate_pair(cache, tiny['collection'][0], 'config')
            for i in range(2):
                path = output/'pairs'/f'{i:04d}.pt'
                saved = torch.load(path, weights_only=True)
                saved['run_config_sha256'] = run.digest(output/'run_config.json')
                torch.save(saved, path)
            predictions = ['2', '2', '9', '2']
            values = {metric: [dict(prediction=p, correct=p=='2') for p in predictions]
                      for metric in ('flexible-extract', 'strict-match')}
            scorer = SimpleNamespace(score=lambda responses, docs: values, provenance={'test': True})
            before = (output/'pairs/0000.pt').read_bytes()
            with patch.object(run.gsm, 'QUESTION_FILE', bank_path), patch.object(run.gsm, 'questions', return_value=questions), patch('rescore_audio_steering.HarnessTaskScorer', return_value=scorer):
                run.report_collection(output)
            result = json.loads((output/'collection-summary.json').read_text())
            self.assertEqual(result['accuracy'][0]['n'], 2)
            self.assertEqual(result['accuracy'][0]['counts']['flexible-numeric'], 1)
            self.assertEqual(result['accuracy'][1]['counts']['flexible-numeric'], 2)
            self.assertEqual(before, (output/'pairs/0000.pt').read_bytes())


if __name__ == '__main__': unittest.main()
