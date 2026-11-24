from copy import deepcopy
import json
from pathlib import Path
import unittest

import compare_bbh5_prompt_subsets as comparison


class SubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference=json.loads((Path(comparison.__file__).parent/'data/bbh5_original_natural_reference.json').read_text())

    def fixture(self):
        old=deepcopy(self.reference)
        prompt='Solve the problem in the audio. End with The answer is (<letter>).'
        rows=[dict(r,prompt=prompt,strength=0.,interventions=[]) for r in old['rows']]
        plan=dict(name='bbh5',prompt=prompt,recipe=deepcopy(old['recipe']),question_order=old['question_ids'],
                  records=[dict(r) for r in rows])
        return old,plan,rows

    def test_exact_original_counts_and_fixed_ids(self):
        old,plan,rows=self.fixture()
        result=comparison.compare(old,plan,rows)
        def get(scope,version,emotion='angry'):
            return next(r for r in result if r['scope']==scope and r['version']==version and
                        r['emotion']==emotion and r['metric']=='flexible-extract')
        all_old=get('all250','original');held_old=get('original-heldout125','original')
        self.assertEqual((all_old['baseline_correct'],all_old['test_correct']),(95,110))
        self.assertEqual((held_old['baseline_correct'],held_old['test_correct']),(42,53))
        self.assertAlmostEqual(all_old['gain_pp'],6.)
        self.assertEqual(get('original-heldout125','current')['n'],125)
        self.assertEqual(get('all250','current-minus-original','neutral')['gain_pp'],0.)

    def test_partial_answers_intersect_fixed_subset_without_resampling(self):
        old,plan,rows=self.fixture();held=set(old['heldout_ids'])
        rows=[r for r in rows if r['question_id'] in held]
        result=comparison.compare(old,plan,rows)
        all_new=next(r for r in result if r['scope']=='all250' and r['version']=='current')
        held_new=next(r for r in result if r['scope']=='original-heldout125' and r['version']=='current')
        self.assertEqual((all_new['n'],all_new['expected']),(125,250))
        self.assertEqual((held_new['n'],held_new['expected']),(125,125))
        self.assertFalse(any(r['scope']=='original-fitting125' and r['version']=='current' for r in result))

    def test_changed_audio_decoding_duplicate_and_steering_rejected(self):
        for change in ('audio','recipe','duplicate','steering'):
            with self.subTest(change=change):
                old,plan,rows=self.fixture()
                if change=='audio':rows[0]['audio_sha256']='wrong'
                elif change=='recipe':plan['recipe']['max_new_tokens']=256
                elif change=='duplicate':rows.append(rows[0])
                else:rows[0]['interventions']=[dict(layer=31,strength=1.)]
                with self.assertRaises(ValueError):comparison.compare(old,plan,rows)


if __name__=='__main__':unittest.main()
