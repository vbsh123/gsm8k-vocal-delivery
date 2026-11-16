import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_multiarith_mean as run


class MultiArithTests(unittest.TestCase):
    def data(self):
        return json.dumps([dict(iIndex=i, sQuestion=f' How many items in question {i % 580}? ',
            lSolutions=[float(i % 580)], lEquations=[f'X={i % 580}']) for i in range(600)]).encode()

    def test_half_split_deduplicates_without_losing_source_ids(self):
        data = self.data()
        with patch.object(run, 'SOURCE_SHA256', hashlib.sha256(data).hexdigest()):
            train, test = run.split_questions(data)
            self.assertEqual((train, test), run.split_questions(data))
        self.assertEqual((len(train), len(test)), (290, 290))
        self.assertFalse({q['question_id'] for q in train} & {q['question_id'] for q in test})
        self.assertFalse({q['question'] for q in train} & {q['question'] for q in test})
        self.assertEqual(sorted(i for q in train+test for i in q['source_ids']), list(range(600)))
        self.assertTrue(all('.' not in q['answer'] for q in train+test))
        self.assertIn('#### <number>', run.PROMPT)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            run.split_questions(data)

    def test_conflicting_duplicate_answers_rejected(self):
        rows = json.loads(self.data()); rows[-1]['lSolutions'] = [-999]
        data = json.dumps(rows).encode()
        with patch.object(run, 'SOURCE_SHA256', hashlib.sha256(data).hexdigest()):
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                run.split_questions(data)

    def test_prepare_uses_own_vector_and_evaluates_only_other_half(self):
        data = self.data()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'audio'; output = root/'out'
            source.mkdir(); (source/'MultiArith-source.json').write_bytes(data)
            def records(source, bank, styles):
                return {(qid, s): dict(question_id=qid, style=s, voice='Jenny', answer=q['answer'])
                        for qid, q in bank.items() for s in styles}
            with patch.object(run, 'SOURCE_SHA256', hashlib.sha256(data).hexdigest()), \
                 patch.object(run.reasoning, 'generate_audio') as tts, \
                 patch.object(run.reasoning, 'source_records', side_effect=records):
                plan = run.prepare(source, output)
                self.assertEqual(plan, run.prepare(source, output))
            self.assertEqual(len(tts.call_args.args[1]), 580)
            self.assertEqual(tts.call_args.args[2], ['neutral', 'angry'])
            self.assertEqual((len(plan['extraction']), len(plan['bank']), len(plan['recordings'])), (290, 290, 870))
            self.assertFalse({p['question_id'] for p in plan['extraction']} & set(plan['bank']))
            self.assertEqual({r['condition'] for r in plan['recordings']}, {'neutral', 'angry', run.reasoning.STEERED})
            self.assertEqual(plan['prompt'], plan['collection_prompt'])
            self.assertEqual(plan['vector'], str((output/'style-vectors.pt').absolute()))
            self.assertEqual(plan['primary_metric'], 'flexible-numeric')
            frozen = json.loads((output/'selection.json').read_text())
            self.assertEqual(frozen['collection'][0]['question_id'], plan['extraction'][0]['question_id'])
            frozen['strength'] = 2
            (output/'selection.json').write_text(json.dumps(frozen))
            with patch.object(run, 'SOURCE_SHA256', hashlib.sha256(data).hexdigest()):
                with self.assertRaisesRegex(ValueError, 'settings changed'):
                    run.prepare_selection(source, output)

    def test_prepare_mode_never_calls_tts_model_or_scorer(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(run.sys, 'argv', ['run_multiarith_mean.py', '--mode', 'prepare', '--output', d]), \
                 patch.object(run, 'prepare_selection', return_value=([{}]*290, [{}]*290)), \
                 patch.object(run.reasoning, 'generate_audio') as tts, \
                 patch.object(run.reasoning, 'scorer_for') as scorer, \
                 patch.object(run.reasoning, 'evaluate') as evaluate:
                run.main()
            tts.assert_not_called(); scorer.assert_not_called(); evaluate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
