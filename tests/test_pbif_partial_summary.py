import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import pbif_emotions as pbif


class PartialSummaryTest(TestCase):
    def test_progress_with_uneven_conditions(self):
        plan = {"bank": {"q1": {}, "q2": {}}, "records": [None] * 8}
        saved = {}

        def add(q, order, style, passes):
            saved[q, f"{order}/{style}"] = dict(response=passes, response_tokens=7)

        with TemporaryDirectory() as tmp, mock.patch.object(pbif, "checks", side_effect=lambda q, v, n: [i < n for i in range(7)]):
            folder = Path(tmp)

            def summary():
                pbif.report(folder, plan, saved, {})
                self.assertEqual(len((folder/"pbif-scores.jsonl").read_text().splitlines()), len(saved))
                return json.loads((folder/"summary.json").read_text())

            add("q1", "tau+1", "neutral", 3)
            self.assertEqual(summary()["cells"]["tau+1/neutral"]["constraints_passed"], 3)
            add("q1", "tau+1", "angry", 7)
            first_pair = summary()
            self.assertEqual(first_pair["matched"], 0)
            self.assertEqual(first_pair["cells"]["tau+1/angry"]["vs_neutral"]["wins"], 1)
            self.assertFalse(first_pair["interactions"]["all_seven"])

            add("q2", "tau-1", "angry", 0)
            self.assertNotIn("vs_neutral", summary()["cells"]["tau-1/angry"])
            add("q1", "tau-1", "neutral", 7)
            add("q1", "tau-1", "angry", 7)
            partial = summary()
            self.assertEqual(partial["matched"], 1)
            self.assertEqual(partial["cells"]["tau-1/angry"]["n"], 2)
            self.assertEqual(partial["cells"]["tau-1/angry"]["vs_neutral"]["pairs"], 1)
            self.assertEqual(partial["interactions"]["all_seven"]["angry"]["gain_pp"], -100)
