import unittest
import json
import tempfile
from pathlib import Path
from contextlib import redirect_stdout
import io
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

    def test_legacy_vector_uses_actual_contributing_pairs_and_angry_field(self):
        import torch
        from show_diffa_forward_counts import original_rows
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            vector=root/'mean/style-vectors.pt'
            vector.parent.mkdir()
            pairs=root/'pairs'
            pairs.mkdir()
            # Legacy exclusion IDs can exceed the pairs that actually fitted the vector.
            torch.save(dict(prompt='Earlier prompt',pairs=[dict(question_id='fit')],
                extraction_question_ids=['fit','excluded']),vector)
            torch.save(dict(pair=dict(question_id='fit'),neutral_forwards=4,angry_forwards=12,matched_forwards=4),pairs/'00000.pt')
            config=root/'run_config.json'
            config.write_text(json.dumps(dict(vector=str(vector))))
            rows=original_rows(config)
            self.assertEqual(len(rows),1)
            self.assertEqual(statistics(rows)['neutral']['mean'],4)
            self.assertEqual(statistics(rows)['emotion']['mean'],12)
            (pairs/'00000.pt').unlink()
            with self.assertRaisesRegex(ValueError,'Missing original'):
                original_rows(config)
