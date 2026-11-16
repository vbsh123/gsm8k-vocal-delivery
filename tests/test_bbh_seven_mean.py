import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import run_bbh_seven_mean as run


class SevenObjectTests(unittest.TestCase):
    def data(self):
        example = dict(input='Seven birds in order.\nOptions:\n'+'\n'.join(f'({c}) Bird {c}' for c in 'ABCDEFG'), target='(G)')
        return json.dumps(dict(examples=[example]*250)).encode()

    def test_disjoint_reproducible_split_all_seven_options_and_prompt(self):
        data = self.data()
        with patch.object(run, 'SOURCE_SHA256', hashlib.sha256(data).hexdigest()):
            train, test = run.split_questions(data)
            self.assertEqual((train, test), run.split_questions(data))
        self.assertEqual((len(train), len(test)), (125, 125))
        self.assertFalse({q['question_id'] for q in train} & {q['question_id'] for q in test})
        self.assertEqual(len({q['question_id'] for q in train+test}), 250)
        self.assertIn('Option G: Bird G', test[0]['question'])
        self.assertEqual(test[0]['answer'], 'G')
        self.assertNotIn('The answer is', test[0]['question'])
        self.assertIn('seven choices A, B, C, D, E, F, G', run.PROMPT)
        self.assertIn('correct option A, B, C, D, E, F, or G', run.PROMPT)

    def test_prepare_collects_own_task_mean_and_only_evaluates_other_half(self):
        data = self.data()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'audio'; source.mkdir(); output = root/'out'; output.mkdir()
            (source/'logical_deduction_seven_objects-source.json').write_bytes(data)
            def records(source, bank, styles):
                return {(q, s): dict(question_id=q, style=s, voice='Jenny', answer='G') for q in bank for s in styles}
            with patch.object(run, 'SOURCE_SHA256', hashlib.sha256(data).hexdigest()), \
                 patch.object(run.reasoning, 'generate_audio') as tts, \
                 patch.object(run.reasoning, 'source_records', side_effect=records):
                plan = run.prepare(source, output)
            self.assertEqual(tts.call_args.args[2], ['neutral', 'angry'])
            self.assertEqual(len(tts.call_args.args[1]), 250)
            self.assertEqual((len(plan['extraction']), len(plan['bank']), len(plan['recordings'])), (125, 125, 375))
            self.assertFalse({p['question_id'] for p in plan['extraction']} & set(plan['bank']))
            self.assertEqual({r['condition'] for r in plan['recordings']}, {'neutral', 'angry', run.reasoning.STEERED})
            self.assertEqual(plan['collection_prompt'], run.PROMPT)
            self.assertEqual((plan['task'], plan['layer'], plan['strength']), ('bbh7', 31, 1))
            self.assertEqual(plan['vector'], str((output/'style-vectors.pt').absolute()))

    def test_real_seven_object_harness_accepts_all_requested_final_answers(self):
        with tempfile.TemporaryDirectory() as d:
            scorer = run.reasoning.scorer_for(dict(task='bbh7'), Path(d))
            docs = [dict(input='Which?\nOptions:\n'+'\n'.join(f'({c}) Bird {c}' for c in 'ABCDEFG'), target=f'({letter})')
                    for letter in 'ABCDEFG']
            values = scorer.score([f'The answer is ({letter}).' for letter in 'ABCDEFG'], docs)
            self.assertTrue(all(row['correct'] for results in values.values() for row in results))
            self.assertEqual(scorer.provenance['task'], 'bbh_cot_zeroshot_logical_deduction_seven_objects')


if __name__ == '__main__':
    unittest.main()
