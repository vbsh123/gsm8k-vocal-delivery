import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import run_multiarith_reasoning_ablation as ablation


class MultiArithAblationTest(unittest.TestCase):
    def test_exact_prompt_and_archived_pairs(self):
        with TemporaryDirectory() as temp:
            plan, questions = ablation.prepare(Path(temp))
        self.assertEqual('Solve the math problem in the audio. ' + ablation.SENTENCE
                         + plan['prompt'].split('audio. ', 1)[1], plan['original_prompt'])
        self.assertEqual(len({(r['question_id'], r['style']) for r in plan['records']}), 580)
        for style, expected in [('neutral', 246), ('angry', 255)]:
            self.assertEqual(sum(q[style]['scores']['flexible-numeric'] for q in questions.values()), expected)

    def test_report_only_matches_completed_pairs_and_resumes(self):
        class Backend:
            def generate(self, record, plan, capture):
                return dict(response='#### '+record['reference_answer'], response_tokens=4, audio_encoding='test')

        with TemporaryDirectory() as temp:
            folder = Path(temp)
            plan, questions = ablation.prepare(folder)
            records = plan['records'][:3]
            ablation.evaluation.generate_dataset(folder, plan, records, Backend())
            ablation.evaluation.generate_dataset(folder, plan, records, Backend())
            self.assertEqual(len(ablation.evaluation.raw_rows(folder, plan)), 3)
            ablation.report(folder, plan, questions)
            summary = json.loads((folder/'summary.json').read_text())
            self.assertEqual(summary['matched'], 1)
            self.assertEqual(len(summary['comparisons']), 8)
            self.assertTrue(all(s['pairs'] == 1 for s in summary['comparisons']))
            self.assertEqual(summary['new_token_lengths']['angry']['mean'], 4)
