import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
import run_kimi_multiarith as run


class KimiMultiArithTests(unittest.TestCase):
    def test_selection_is_disjoint_complete_and_frozen_before_outcomes(self):
        from tests.test_multiarith_mean import MultiArithTests
        data = MultiArithTests().data()
        with tempfile.TemporaryDirectory() as d:
            source = Path(d)/'audio'; source.mkdir()
            output = Path(d)/'out'; output.mkdir()
            (source/'MultiArith-source.json').write_bytes(data)
            with patch.object(run.multiarith, 'SOURCE_SHA256', hashlib.sha256(data).hexdigest()):
                plan = run.select(source, output)
                self.assertEqual(plan, run.select(source, output))
                self.assertEqual((len(plan['collection']), len(plan['test'])), (290, 290))
                self.assertFalse({q['question'] for q in plan['collection']} & {q['question'] for q in plan['test']})
                self.assertEqual((plan['layer'], plan['group'], plan['strength']), (17, 'answer', 1.))
                self.assertIn('#### <number>', plan['prompt'])
                changed = dict(plan, layer=18)
                (output/'selection.json').write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'Settings changed'):
                    run.select(source, output)

    def fixture(self, root):
        def pair(qid):
            return dict(question_id=qid, voice='Jenny', **{s: dict(question_id=qid,
                voice='Jenny', style=s, audio_file=f'{qid}-{s}', audio_sha256=f'{qid}-{s}',
                reference_answer='2') for s in ('neutral', 'angry')})
        return dict(collection=[pair('train0'), pair('train1')], test=[pair('test0')],
                    prompt=run.multiarith.PROMPT, model='Kimi', revision='pinned',
                    vector=str(root/'style-vectors.pt'))

    def test_equal_pair_mean_small_caches_and_resume_preserves_vector(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = self.fixture(root)
            tokenizer = SimpleNamespace(encode=lambda text, *, bos, eos: list(text))
            model = SimpleNamespace(prompt_manager=SimpleNamespace(text_tokenizer=tokenizer))
            def trace(model, row, prompt):
                # Deliberately invalid collection responses must still contribute.
                amount = (2 if row['question_id'] == 'train0' else 6) if row['style'] == 'angry' else 0
                return dict(response='no numeric answer', means=torch.full((28, 3), float(amount)),
                            audio_means=torch.full((28, 3), float(amount*2)), calls=[0])
            with patch.object(run.kimi, 'digest', side_effect=lambda p: str(p)), \
                 patch.object(run.kimi, 'trace_generation', side_effect=trace) as generation:
                vector = run.collect(plan, root, model)
                digest = hashlib.sha256((root/'style-vectors.pt').read_bytes()).hexdigest()
                run.collect(plan, root, model)
                self.assertEqual(generation.call_count, 4)
                self.assertEqual(digest, hashlib.sha256((root/'style-vectors.pt').read_bytes()).hexdigest())
            self.assertTrue(torch.equal(vector['answer_direction'], torch.full((3,), 4.)))
            self.assertTrue(torch.equal(vector['audio_direction'], torch.full((3,), 8.)))
            cached = torch.load(root/'pairs/00000.pt', weights_only=True)
            self.assertNotIn('inputs', cached)
            self.assertEqual(cached['answer_difference'].shape, (3,))

    def test_evaluate_all_test_conditions_native_prompt_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = self.fixture(root)
            vector = dict(collection_question_ids=['train0', 'train1'], answer_direction=torch.ones(3))
            tokenizer = SimpleNamespace(encode=lambda text, *, bos, eos: list(text))
            model = SimpleNamespace(prompt_manager=SimpleNamespace(text_tokenizer=tokenizer))
            with patch.object(run.kimi, 'digest', side_effect=lambda p: str(p)), \
                 patch.object(run.kimi, 'generate', return_value='#### 2') as plain, \
                 patch.object(run.kimi, 'generate_steered', return_value='#### 2') as steered:
                run.evaluate(plan, root, model, vector)
                run.evaluate(plan, root, model, vector)
            self.assertEqual(plain.call_count, 2)
            self.assertEqual(steered.call_count, 1)
            args = steered.call_args.args
            self.assertEqual(args[2:5], (plan['prompt'], 17, 'answer'))
            rows = run.read_rows(root/'results.jsonl')
            self.assertEqual(len(rows), 3)
            self.assertEqual({r['question_id'] for r in rows}, {'test0'})
            self.assertEqual({r['condition'] for r in rows}, {'neutral', 'angry', run.STEERED})
            self.assertTrue(all(r['seed'] == 42 and r['max_new_tokens'] == 512 for r in rows))

    def test_installed_harness_report_keeps_numeric_and_format_separate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); plan = self.fixture(root)
            plan.update(bank={'test0': dict(question='One plus one?', answer='2')},
                        selection={'primary_comparison': 'steered versus neutral'}, scoring='Audio adaptation.')
            (root/'run_config.json').write_text(json.dumps(plan))
            responses = {'neutral': 'The answer is 2.', 'angry': '#### 2', run.STEERED: '#### 3'}
            rows = [dict(question_id='test0', voice='Jenny', condition=c, response=t,
                         reference_answer='2', prompt=plan['prompt'], response_tokens=5)
                    for c, t in responses.items()]
            (root/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            original = (root/'results.jsonl').read_bytes()
            run.report(root)
            summary = json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['matched_all_three'], 1)
            counts = {r['condition']: r['correct'] for r in summary['accuracy']}
            self.assertEqual(counts['neutral']['flexible-numeric'], 1)
            self.assertEqual(counts['neutral']['strict-match'], 0)
            self.assertEqual(counts['angry']['strict-match'], 1)
            self.assertEqual(counts[run.STEERED]['format-compliant'], 1)
            self.assertEqual(counts[run.STEERED]['flexible-numeric'], 0)
            self.assertEqual(original, (root/'results.jsonl').read_bytes())


if __name__ == '__main__':
    unittest.main()
