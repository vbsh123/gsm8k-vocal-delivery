import unittest
from show_diffa_forward_counts import statistics


class ForwardCountTests(unittest.TestCase):
    def test_matched_fraction_is_per_pair_and_neutral_counts_are_independent(self):
        rows=[dict(neutral_forwards=2,emotion_forwards=10,matched_forwards=2),
              dict(neutral_forwards=8,emotion_forwards=4,matched_forwards=4)]
        stats=statistics(rows)
        self.assertEqual(stats['neutral']['mean'],5)
        self.assertEqual(stats['emotion']['mean'],7)
        self.assertEqual(stats['angry_longer'],1)
        self.assertEqual(stats['actual_angry_retained_mean_pct'],60)
        self.assertEqual(stats['hypothetical_matched_angry_retained_mean_pct'],60)

    def test_all_forward_mode_keeps_every_forward_and_reports_counterfactual(self):
        stats=statistics([dict(neutral_forwards=2,emotion_forwards=10,matched_forwards=None)])
        self.assertEqual(stats['actual_angry_retained_mean_pct'],100)
        self.assertEqual(stats['hypothetical_matched_angry_retained_mean_pct'],20)
        self.assertIsNone(statistics([]))
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            statistics([dict(neutral_forwards=2,emotion_forwards=10,matched_forwards=3)])
        with self.assertRaisesRegex(ValueError,'positive'):
            statistics([dict(neutral_forwards=0,emotion_forwards=10,matched_forwards=None)])
