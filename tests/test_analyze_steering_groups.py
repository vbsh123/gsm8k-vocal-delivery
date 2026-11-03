"""Verify input-label grouping and matched baseline/steering comparisons."""

import unittest

from analyze_steering_groups import build_groups, flexible_correct, paired_stats


class SteeringGroupTests(unittest.TestCase):
    def test_group_membership_uses_each_classifier_without_intersection(self):
        ids = ["a", "b", "c", "d"]
        diffa = {"a": {"predicted_style": "yes"}, "b": {"predicted_style": "no"},
                 "c": {"predicted_style": "yes"}}
        external = {"a": {"binary_prediction": "emotional"},
                    "b": {"binary_prediction": "neutral"},
                    "c": {"binary_prediction": "unresolved"}}
        groups = build_groups(ids, diffa, external)
        self.assertEqual(groups["DIFFA: neutral"], ["a", "c"])
        self.assertEqual(groups["emotion2vec: neutral"], ["b"])
        self.assertEqual(groups["emotion2vec: unresolved"], ["c"])
        self.assertEqual(groups["emotion2vec: missing/invalid"], ["d"])
        self.assertEqual(groups["DIFFA: missing/invalid"], ["d"])

    def test_same_clip_ids_are_used_for_both_accuracies(self):
        baseline = {"a": {"correct": False}, "b": {"correct": True},
                    "outside_group": {"correct": True}}
        steered = {"b": {"correct": True}, "a": {"correct": True},
                   "outside_group": {"correct": False}}
        stats = paired_stats(["a", "b"], baseline, steered, lambda r: r["correct"])
        self.assertEqual((stats["n"], stats["steered"], stats["baseline"]), (2, 2, 1))
        self.assertEqual((stats["wins"], stats["losses"]), (1, 0))
        self.assertEqual(stats["gain_points"], 50)
        self.assertEqual(stats["mcnemar_p"], 1)
        self.assertIsNone(paired_stats([], baseline, steered, lambda r: r["correct"]))

    def test_non_strict_uses_last_number_without_reference_guided_extraction(self):
        self.assertTrue(flexible_correct({"response": "12.", "reference_answer": "12"}))
        self.assertFalse(flexible_correct({"response": "Each costs $3, totaling $135.",
                                           "reference_answer": "3"}))
        self.assertFalse(flexible_correct({"response": "No answer", "reference_answer": "3"}))


if __name__ == "__main__":
    unittest.main()
