import unittest

from scoring import extract_answer, score_response


class ScoringTests(unittest.TestCase):
    def test_markdown_and_numeric_representations(self):
        for text, expected in [
            ("**Final answer:** 5", "5"),
            ("**Final answer: 5**", "5"),
            ("Final answer:\n5", "5"),
            ("Final answer: $1,234.00 dollars", "1234"),
            (r"Final answer: \boxed{\frac{1}{2}}", "0.5"),
            ("Final answer: −2", "-2"),
            ("Final answer: 1e3", "1000"),
        ]:
            with self.subTest(text=text):
                self.assertTrue(score_response(text, expected)["correct"])

    def test_wrong_answer_stays_wrong_after_markdown_fix(self):
        score = score_response("**Final answer:** 40", "70")
        self.assertEqual(score["predicted_answer"], "40")
        self.assertFalse(score["correct"])
        self.assertFalse(score["parse_failed"])

    def test_component_counts_and_expressions_are_not_silently_accepted(self):
        for text in ("Final answer: 4 blue, 6 red", "Final answer: 4 + 6",
                     "Final answer: 4 = ten", "Final answer: 5/0"):
            with self.subTest(text=text):
                self.assertIsNone(extract_answer(text)[0])

    def test_reasoning_number_is_not_used_as_a_fallback(self):
        self.assertTrue(score_response("20 + 8 + 20 = 48 legs\nFinal answer", "48")["parse_failed"])

    def test_last_explicit_final_answer_wins_without_reference_matching(self):
        text = "Final answer: 5\nCorrection. Final answer: 7"
        self.assertEqual(extract_answer(text)[0], "7")
        self.assertFalse(score_response(text, "5")["correct"])
