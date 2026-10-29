import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from crop_questions import alignment_words, crop_interval, question_parts


class CropTests(unittest.TestCase):
    def spans(self, words):
        # Each word occupies the middle 60% of a one-second interval.
        return [[SimpleNamespace(start=i * 10 + 2, end=i * 10 + 8)] for i in range(words)]

    def test_all_templates_keep_question_words_and_cut_only_at_remark_sides(self):
        templates = json.loads((Path(__file__).resolve().parents[1] / "configs/remarks.json").read_text())
        question = "Maya has twelve apples. How many remain?"
        for style, variants in templates.items():
            for template in variants:
                with self.subTest(style=style, template=template):
                    full_text = template.format(question=question)
                    parts = question_parts(full_text, question)
                    self.assertEqual(parts[1], alignment_words(question))
                    n = sum(map(len, parts))
                    start, end = crop_interval(parts, self.spans(n), 0.1, n, 0.04)
                    first, last = len(parts[0]), len(parts[0]) + len(parts[1]) - 1
                    self.assertLessEqual(start, first + 0.2)
                    self.assertGreaterEqual(end, last + 0.8)
                    if parts[0]:
                        self.assertGreater(start, first - 0.2)
                    else:
                        self.assertEqual(start, 0)
                    if parts[2]:
                        self.assertLess(end, last + 1.2)
                    else:
                        self.assertEqual(end, n)

    def test_large_padding_stops_at_midpoints_between_question_and_remarks(self):
        start, end = crop_interval([["please"], ["maya", "runs"], ["thanks"]],
                                   self.spans(4), 0.1, 4, 10)
        self.assertAlmostEqual(start, 1)
        self.assertAlmostEqual(end, 3)

    def test_numbers_are_wildcards_and_contractions_are_preserved(self):
        self.assertEqual(alignment_words("What's José’s $1,200.50 total?"),
                         ["what's", "jose's", "*", "total"])

    def test_missing_or_repeated_question_is_rejected(self):
        for text in ("Maya has apples.", "Question? Question?"):
            with self.assertRaises(ValueError):
                question_parts(text, "Question?")

    def test_incomplete_alignment_is_rejected(self):
        with self.assertRaises(ValueError):
            crop_interval([["please"], ["help"], []], self.spans(1), 0.1, 2, 0.04)


if __name__ == "__main__":
    unittest.main()
