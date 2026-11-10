import unittest

from build_jenny_report import agrees, paired_comparisons, steering_table, steering_rows, render_report


class MatchedReportTests(unittest.TestCase):
    def test_all_conditions_and_mixed_strength_combinations_are_visible_without_javascript(self):
        experiment = dict(id='recovery-test', title='Repair <test>', source='Same selected questions.', layers={
            '12': dict(n=10, baseline=4, points=[dict(group='acoustic',strength=1,correct=6,n=10,wins=3,losses=1,p=.625)]),
            '5 + 11': dict(n=6, baseline=0, points=[dict(group='combo',strength=1,correct=1,n=6,wins=1,losses=0,p=1,
                interventions=[dict(layer=5,group='acoustic',strength=1),dict(layer=11,group='answer',strength=.5)])])})
        rows = steering_rows(experiment)
        self.assertEqual([r['condition'] for r in rows], ['L5 acoustic ×1 + L11 answer ×0.5', 'L12 acoustic ×1'])
        html = steering_table(experiment)
        self.assertNotIn('<select', html)
        self.assertIn('Repair &lt;test&gt;', html)
        self.assertIn('L5 acoustic ×1 + L11 answer ×0.5', html)
        self.assertIn('<td>4/10</td><td>40.0%</td><td>6/10</td><td>60.0%</td>', html)
        self.assertIn('>+20.0</td>', html)
        self.assertIn('Previously wrong answers repaired', html)

    def test_render_preserves_existing_speed_data(self):
        speed = dict(source='saved measurements', points=[])
        html = render_report([], dict(models=[],scorer='test',filters=[]), speed)
        self.assertIn('"source": "saved measurements"', html)
        self.assertNotIn('__STEERING_TABLES__', html)
        self.assertNotIn('class="layer"', html)
        self.assertIn('L23 answer ×1.25', html)

    def test_union_is_per_recording_and_intersection_requires_both_classifiers(self):
        tasks = {('q', 'v', s): dict(question_id='q', voice='v', style=s)
                 for s in ('neutral', 'angry')}
        answers = {key: dict(ok=key[2] == 'angry', reference_answer='1') for key in tasks}
        labels = {('q', 'v', 'neutral'): dict(emotion2vec='neutral', sensevoice='angry'),
                  ('q', 'v', 'angry'): dict(emotion2vec='neutral', sensevoice='angry')}
        rows = {r['filter']: r for r in paired_comparisons(tasks, answers, labels)
                if (r['reference'], r['test']) == ('neutral', 'angry')}
        self.assertEqual((rows['unfiltered']['n'], rows['unfiltered']['wins']), (1, 1))
        self.assertEqual(rows['emotion2vec']['n'], 0)
        self.assertEqual(rows['both']['n'], 0)
        self.assertEqual((rows['either']['n'], rows['either']['wins']), (1, 1))

    def test_matching_excludes_different_voices_missing_answers_and_wrong_neutral(self):
        keys = [('q1', 'v1', 'neutral'), ('q1', 'v2', 'angry'),
                ('q2', 'v1', 'neutral'), ('q2', 'v1', 'angry'),
                ('q3', 'v1', 'neutral'), ('q3', 'v1', 'angry')]
        tasks = {k: dict(question_id=k[0], voice=k[1], style=k[2]) for k in keys}
        answers = {k: dict(ok=True, reference_answer='1') for k in keys if k != ('q3', 'v1', 'angry')}
        labels = {k: dict(emotion2vec='angry', sensevoice='angry') for k in keys}
        rows = {r['filter']: r for r in paired_comparisons(tasks, answers, labels)
                if (r['reference'], r['test']) == ('neutral', 'angry')}
        self.assertEqual(rows['unfiltered']['pair_ids'], [['q2', 'v1']])
        self.assertEqual(rows['either']['n'], 0)
        self.assertIsNone(rows['either']['p'])

    def test_missing_labels_do_not_pass_and_contradictory_gold_is_rejected(self):
        self.assertFalse(agrees('neutral', {}, 'both'))
        self.assertFalse(agrees('neutral', {}, 'either'))
        self.assertTrue(agrees('neutral', {}, 'unfiltered'))
        tasks = {('q', 'v', s): dict(question_id='q', voice='v', style=s)
                 for s in ('neutral', 'angry')}
        answers = {key: dict(ok=True, reference_answer='1' if key[2] == 'angry' else '2') for key in tasks}
        with self.assertRaises(ValueError):
            paired_comparisons(tasks, answers, {})

    def test_mcnemar_counts_use_only_matched_questions(self):
        tasks, answers = {}, {}
        for i in range(6):
            for style in ('neutral', 'sad'):
                key = (str(i), 'v', style)
                tasks[key] = dict(question_id=str(i), voice='v', style=style)
                answers[key] = dict(ok=style == 'sad', reference_answer='1')
        row = next(r for r in paired_comparisons(tasks, answers, {})
                   if (r['filter'], r['reference'], r['test']) == ('unfiltered', 'neutral', 'sad'))
        self.assertEqual((row['n'], row['wins'], row['losses']), (6, 6, 0))
        self.assertEqual(row['p'], .03125)


if __name__ == '__main__':
    unittest.main()
