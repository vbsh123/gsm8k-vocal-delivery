from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import audit_harness_api as audit
from rescore_audio_steering import HarnessTaskScorer


def rows(path, entries):
    path.write_text(''.join(json.dumps(r)+'\n' for r in entries))


@unittest.skipUnless(importlib.util.find_spec('lm_eval'), 'Needs installed lm_eval task API')
class HarnessAPITests(unittest.TestCase):
    def test_standard_task_methods_used_without_downloading_dataset_or_running_model(self):
        with tempfile.TemporaryDirectory() as d:
            scorer = HarnessTaskScorer('gsm8k', Path(d))
            with patch('datasets.load_dataset', side_effect=AssertionError('Saved-answer scoring must not download a dataset')):
                result = scorer.score(['#### 5', 'The result is 5'],
                    [dict(question='2+3?', answer='5')]*2)
            self.assertEqual([r['correct'] for r in result['strict-match']], [True, False])
            self.assertEqual([r['correct'] for r in result['flexible-extract']], [True, True])
            self.assertIn('process_results', scorer.provenance['api'])

    def test_svamp_api_rescore_preserves_answers_and_prior_summary(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); run = root/'run'; run.mkdir(); source = root/'source'; source.mkdir()
            destination = root/'audit'
            data = json.dumps([dict(ID=str(i), Body=f'Body {i}', Question='How many?',
                Answer=5, Equation='2+3', Type='Addition') for i in range(1000)]).encode()
            (source/'SVAMP-source.json').write_bytes(data)
            questions = audit.select_questions(data, 500); rows(source/'questions.jsonl', questions)
            qid = questions[0]['question_id']
            answers = [dict(question_id=qid, voice='Jenny', condition=c, response=response, reference_answer='5',
                harness_strict=correct, harness_flexible=correct, flexible_numeric=correct)
                for c, response, correct in [('neutral', '#### 4', False), ('angry', '#### 5', True)]]
            rows(run/'results.jsonl', answers)
            old = dict(accuracy=[dict(condition=c, n=1, correct=dict(harness_strict=int(ok),
                harness_flexible=int(ok), flexible_numeric=int(ok))) for c, ok in [('neutral', False), ('angry', True)]])
            (run/'summary.json').write_text(json.dumps(old))
            before = (run/'results.jsonl').read_bytes(), (run/'summary.json').read_bytes()
            with patch.object(audit, 'SVAMP_SHA', hashlib.sha256(data).hexdigest()):
                audit.audit_svamp(run, source, destination)
            self.assertEqual(before, ((run/'results.jsonl').read_bytes(), (run/'summary.json').read_bytes()))
            self.assertEqual((destination/'disagreements.jsonl').read_text(), '')
            self.assertTrue(all(c['same_correct_count'] for c in json.loads((destination/'comparison.json').read_text())))
            result = json.loads((destination/'summary.json').read_text())
            self.assertTrue(all(r['wins'] == 1 and r['losses'] == 0 for r in result['paired'][:3]))

    def test_bbh_steering_rescore_uses_actual_api_and_separate_destination(self):
        from steer_bbh_mean import summarize
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            root = Path(d); run = root/'run'; run.mkdir(); source = root/'source'; source.mkdir()
            (run/'split.json').write_text(json.dumps(dict(test=[dict(question_id='q')], extraction=[dict(question_id='train')])))
            question = dict(question_id='q', original_input='Which?\nOptions:\n(A) red\n(B) blue\n(C) green',
                original_target='(B)', answer='B')
            rows(source/'questions.jsonl', [question])
            rows(run/'results.jsonl', [dict(question_id='q', voice='Jenny', condition=c, reference_answer='B', response=response)
                for c, response in [('neutral', 'The answer is (A).'), ('angry', 'The answer is (B).'),
                                    ('neutral-BBH-L31-answer-a1', 'The answer is (B).')]])
            (run/'summary.json').write_text('original summary')
            before = (run/'results.jsonl').read_bytes(), (run/'summary.json').read_bytes()
            result = summarize(SimpleNamespace(output=run, source=source), destination=root/'audit')
            self.assertEqual(before, ((run/'results.jsonl').read_bytes(), (run/'summary.json').read_bytes()))
            self.assertTrue(all(r['wins'] == 1 and r['losses'] == 0 for r in result['paired']))
            self.assertTrue((root/'audit/rescored.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
