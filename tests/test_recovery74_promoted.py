import unittest

from steer_jenny_recovery74_promoted import select_conditions


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.recordings = [dict(clip_id=str(i), question_id=str(i), style='neutral') for i in range(74)]
        self.conditions = [dict(name='baseline', interventions=[])] + [
            dict(name=f'L{layer}-acoustic-a1.25', interventions=[dict(layer=layer, group='acoustic', strength=1.25)])
            for layer in range(3)]
        self.source = dict(recordings=self.recordings)
        self.screen = dict(recordings=self.recordings[:10], settings=dict(conditions=self.conditions))
        self.rows = []
        for i in range(10):
            for j, c in enumerate(self.conditions):
                ok = j > 0 and i < [0, 3, 4, 5][j]
                self.rows.append(dict(clip_id=str(i), condition=c['name'],
                                      response='Final answer: '+('1' if ok else '0'), reference_answer='1'))

    def test_threshold_is_inclusive_and_all74_are_preserved(self):
        conditions, evidence = select_conditions(self.source, self.screen, self.rows)
        self.assertEqual([c['name'] for c in conditions], ['baseline', 'L1-acoustic-a1.25', 'L2-acoustic-a1.25'])
        self.assertEqual([r['recovered'] for r in evidence], [4, 5])
        self.assertEqual(len(self.source['recordings']), 74)

    def test_incomplete_screen_and_wrong_cohort_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            select_conditions(self.source, self.screen, self.rows[:-1])
        self.screen['recordings'] = [dict(r) for r in self.screen['recordings']]
        self.screen['recordings'][0]['question_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'differs'):
            select_conditions(self.source, self.screen, self.rows)

    def test_screen_baseline_must_still_fail(self):
        self.rows[0]['response'] = 'Final answer: 1'
        with self.assertRaisesRegex(ValueError, 'baseline failures'):
            select_conditions(self.source, self.screen, self.rows)


if __name__ == '__main__':
    unittest.main()
