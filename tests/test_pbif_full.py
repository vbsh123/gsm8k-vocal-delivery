import unittest

from pbif_full import describe, new_questions


class FullPBIF(unittest.TestCase):
    def test_same_weight_for_each_base_request_after_pairing(self):
        groups = {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}
        scores = {}
        for q in groups:
            scores[q, "tau+1", "neutral"] = 0
            if q != "a3":
                scores[q, "tau+1", "angry"] = int(q == "b1")
        stats = describe(groups, scores, groups, "tau+1")
        self.assertEqual((stats["rows"], stats["base_requests"]), (3, 2))
        self.assertEqual(stats["gain_pp"], 50)

    def test_spoken_audio_only_for_new_rows_and_both_orders(self):
        row = {"prompts": {"tau+1": "First order.", "tau-1": "Second order."}}
        questions = new_questions({"pilot": row, "new": row}, {"pilot": row})
        self.assertEqual([q["question"] for q in questions.values()], list(row["prompts"].values()))
        self.assertEqual(set(questions), {"new/tau+1", "new/tau-1"})


if __name__ == "__main__":
    unittest.main()
