from contextlib import redirect_stdout
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import collect_instruction_controls
from analyze_phi4_layer_drift import analyze
from tests.test_reasoning_geometry_audit import GeometryAuditTests


class PhiLayerDriftTests(unittest.TestCase):
    def test_saved_states_relative_change_cosine_and_fixed_cohort(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            source = Path(d)/'source'; source.mkdir(); output = Path(d)/'analysis'
            shape = GeometryAuditTests().fixture(source, 'phi4')
            with patch.object(collect_instruction_controls, 'state_shape', return_value=shape):
                analyze(source, output)
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(len(summary['questions']), 100)
            self.assertEqual(len(summary['rows']), 64)
            row = summary['rows'][0]
            expected = sum(100*4/(math.sqrt(3)*i) for i in range(1, 100))/99
            self.assertAlmostEqual(row['mean_relative_change_pct'], expected)
            self.assertAlmostEqual(row['median_relative_change_pct'], 100*4/(math.sqrt(3)*50))
            self.assertEqual(row['pairs'], 99)  # Synthetic first neutral has zero norm.
            self.assertAlmostEqual(row['shift_direction_consistency'], 1.)
            with (output/'pair-metrics.csv').open() as stream:
                pairs = list(csv.DictReader(stream))
            row = next(r for r in pairs if r['question_id'] == 'q4' and r['group'] == 'answer' and r['layer'] == '0')
            self.assertAlmostEqual(float(row['representation_cosine']), math.sqrt(2/3))
            # Missing one original capture must not silently change the selected cohort.
            (source/'states/000-angry-reasoning.pt').unlink()
            with patch.object(collect_instruction_controls, 'state_shape', return_value=shape):
                with self.assertRaisesRegex(ValueError, 'all100'): analyze(source, output)


if __name__ == '__main__': unittest.main()
