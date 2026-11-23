import unittest

from sample_diffa_steering_failures import cases, lengths, render


class FailureSampleTests(unittest.TestCase):
    def fixture(self):
        config = dict(target=dict(prompt='same', records=[dict(question_id=q, voice='Jenny', audio_sha256=q)
                      for q in ('loss', 'bothwrong', 'win', 'format')],
                      bank={q: dict(question='Question '+q, answer='2') for q in ('loss', 'bothwrong', 'win', 'format')}),
                      source=dict(vector_sha256='v'))
        rows = []
        for q, before, after in [('loss', True, False), ('bothwrong', False, False), ('win', False, True), ('format', True, True)]:
            for strength, correct in [(0., before), (1., after)]:
                rows.append(dict(question_id=q, voice='Jenny', audio_sha256=q, prompt='same', strength=strength,
                    vector_sha256='v' if strength else None, response='Full response', reference_answer='2',
                    response_tokens=30 if strength else 5, predictions={'flexible-numeric': '2'},
                    metrics={'flexible-numeric': correct, 'format-compliant': not (q == 'format' and strength)}))
        return config, rows

    def test_sample_only_new_numeric_errors_with_exact_paired_baseline(self):
        config, rows = self.fixture()
        sample, paired = cases(config, rows)
        self.assertEqual(paired, 4)
        self.assertEqual([r['question_id'] for r in sample], ['loss'])
        lengths(sample)
        self.assertEqual(sample[0]['extra_tokens'], 25)
        text = render(sample, 'Heading')
        self.assertIn('Question loss', text)
        self.assertIn('Full response', text)
        self.assertIn('tokens=30', text)
        self.assertEqual(len(cases(config, rows, kind='errors')[0]), 2)
        self.assertEqual(cases(config, rows, kind='format')[0][0]['question_id'], 'format')

    def test_mismatched_vector_or_missing_scores_fail_instead_of_mixing_runs(self):
        config, rows = self.fixture()
        rows[1]['vector_sha256'] = 'another vector'
        with self.assertRaises(ValueError):
            cases(config, rows)
        config, rows = self.fixture()
        with self.assertRaisesRegex(ValueError, 'Metric'):
            cases(config, rows, metric='absent')


if __name__ == '__main__':
    unittest.main()
