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

    def test_three_asks_on_original_recordings(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            plan, questions = ablation.prepare(folder, three_asks=True)
            self.assertEqual(plan['prompt'].replace(ablation.THIRD_ASK + ' ', ''), plan['original_prompt'])
            self.assertEqual(len(plan['records']), 580)
            self.assertEqual(sum(v['angry']['scores']['written-operation'] for v in questions.values()), 205)
            with self.assertRaises(ValueError):
                ablation.prepare(folder)

            class Backend:
                def generate(self, record, plan, capture):
                    opening = 'Ready. ' if record['style'] == 'angry' else ''
                    return dict(response=opening+'2 + 1 = 3.\n#### '+record['reference_answer'],
                                response_tokens=12, audio_encoding='test')

            ablation.evaluation.generate_dataset(folder, plan, plan['records'][:5], Backend())
            ablation.report(folder, plan, questions)
            stats = json.loads((folder/'summary.json').read_text())
            joint = next(c for c in stats['comparisons'] if c['prompt']=='three asks' and c['metric']=='all-three-observed')
            self.assertEqual((stats['matched'], joint['baseline_correct'], joint['steered_correct']), (2, 0, 2))
            self.assertEqual(len(stats['changes']), 8)

    def test_instruction_measures_are_independent(self):
        from multiarith_instruction_checks import checks
        self.assertFalse(checks('Ready.\n#### 2')['written-operation'])
        self.assertFalse(checks('Already done. 1+1=2\n#### 2')['ready-opening'])
        self.assertFalse(checks('Ready. 1+1=2\n#### 2 dollars')['exact-final-line'])
        self.assertTrue(checks('  ready! 1+1=5\n#### 5  ')['all-three-observed'])
