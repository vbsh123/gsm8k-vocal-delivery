import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import diffa_gsm_final_answer as run


class FinalAnswerTests(unittest.TestCase):
    def test_prepare_keeps_original_and_changes_all_delivery_prompts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'source'; source.mkdir()
            old = root/'old.json'
            old.write_text(json.dumps(dict(recordings=[dict(question_id='q', condition='angry', audio_file='angry.wav')])))
            original = dict(name='gsm8k', collection_prompt='Final answer: <number>', prompt='#### <number>',
                            test_ids=['q'], bank={'q': {'answer': '2'}},
                            target=dict(prompt='#### <number>', config_file=str(old)),
                            recordings=[dict(question_id='q', condition=c) for c in ['neutral', 'happy']])
            path = source/'run_config.json'; path.write_text(json.dumps(original)); before = path.read_bytes()
            plan = run.prepare(source, root/'new')
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(plan['prompt'], plan['collection_prompt'])
            self.assertEqual(plan['target']['prompt'], plan['collection_prompt'])
            self.assertEqual([r['condition'] for r in plan['recordings']], ['neutral', 'happy', 'angry'])
            self.assertEqual(plan['recordings'][-1]['reference_answer'], '2')
            self.assertEqual(run.prepare(source, root/'new'), plan)

    def test_original_scorer_and_common_questions_including_steering(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [dict(question_id='q', condition=c, response=text, reference_answer='2', response_tokens=3)
                    for c, text in [('neutral', 'Final answer: 1'), ('happy', 'Final answer: 2'),
                                    ('angry', 'Final answer: 2'), ('neutral-L31-answer-a1', 'Final answer: 2')]]
            rows.append(dict(question_id='extra', condition='happy', response='Final answer: 2', reference_answer='2'))
            with patch('builtins.print'):
                run.report(rows, Path(d), 2)
            saved = json.loads((Path(d)/'summary.json').read_text())
            self.assertEqual(saved['matched'], 1)
            self.assertEqual(saved['conditions']['neutral']['correct'], 0)
            for label in ['happy', 'angry', 'neutral-L31-answer-a1']:
                self.assertEqual(saved['conditions'][label]['correct'], 1)
                self.assertEqual(saved['comparisons'][label]['gain_pp'], 100)


if __name__ == '__main__':
    unittest.main()
