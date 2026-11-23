import unittest

from phi4_compare_delivery_cohorts import compare,matched_deliveries


class CohortComparisonTests(unittest.TestCase):
    def test_intersection_is_based_on_ids_and_excludes_fresh_collection_and_steering(self):
        def row(q,s,correct,split='test',strength=0.):
            return dict(question_id=q,voice='Jenny',condition=s,response=str(correct),correct=correct,
                        split=split,strength=strength,audio_sha256=q+s,prompt='same',max_new_tokens=512)
        old=matched_deliveries([row(q,s,c) for q,cs in [('a',(False,True)),('b',(True,False)),('c',(False,True))]
                                for s,c in zip(('neutral','angry'),cs)])
        fresh=matched_deliveries([row('a','neutral',True),row('a','angry',True),
                                 row('b','neutral',False),row('b','angry',True),
                                 row('c','neutral',True,'collection'),row('c','angry',True,'collection'),
                                 row('d','neutral',True,strength=1.),row('d','angry',True)],heldout=True)
        def score(rows):
            return [dict(r,metrics={'numeric':r['correct']}) for r in rows]
        def paired(values):
            return dict(pairs=len(values),gain_pp=100*sum(int(b)-int(a) for a,b in values)/len(values))
        result=compare(old,fresh,score,paired)
        self.assertEqual(result['common_question_voice_ids'],[('a','Jenny'),('b','Jenny')])
        self.assertEqual(result['old_questions_missing_fresh_heldout'],1)
        self.assertEqual(result['cohorts']['old-common']['numeric']['gain_pp'],0.)
        self.assertEqual(result['cohorts']['fresh-common']['numeric']['gain_pp'],50.)
        self.assertEqual(result['cross_run']['neutral']['same_prompt'],2)

    def test_conflicting_duplicate_responses_are_rejected(self):
        row=dict(question_id='a',voice='Jenny',condition='neutral',response='1')
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            matched_deliveries([row,dict(row,response='2')])


if __name__=='__main__':
    unittest.main()
