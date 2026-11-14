import unittest

from audit_commonsense_scoring import relaxed_choice


class RelaxedChoiceTests(unittest.TestCase):
    choices=[dict(label=k,text=v) for k,v in zip('ABCDE',['shop','library','bank','park','station'])]

    def test_clear_format_variations_without_gold_matching(self):
        for response in ['Final answer: B','Final answer: B. library','The correct answer is B because books are there.',
                         'Answer: library','Reasoning.\nB','Final answer: (B)',r'Final answer: \boxed{B}']:
            with self.subTest(response=response):self.assertEqual(relaxed_choice(response,self.choices),'B')
        self.assertEqual(relaxed_choice('Final answer: C. bank',self.choices),'C')

    def test_ambiguous_and_conflicting_answers_are_rejected(self):
        for response in ['Final answer: B or C','Final answer: B/C','Final answer: B and C',
                         'Final answer: A. library','B appears somewhere in the reasoning, probably.',
                         'Reasoning:\nA person walks outside.']:
            with self.subTest(response=response):self.assertIsNone(relaxed_choice(response,self.choices))

    def test_last_explicit_answer_wins_and_duplicate_option_text_is_ambiguous(self):
        self.assertEqual(relaxed_choice('Final answer: B\nCorrection. Final answer: C',self.choices),'C')
        choices=self.choices+[dict(label='F',text='library')]
        self.assertIsNone(relaxed_choice('Final answer: library',choices))

    def test_sentence_article_is_not_an_alternative_choice(self):
        for response in ['The correct answer is B. A library is where books are kept.',
                         'Answer: B, a library is where books are kept.']:
            self.assertEqual(relaxed_choice(response,self.choices),'B')
        for response in ['Final answer: B. C','Final answer: B, C','Final answer: B: C']:
            self.assertIsNone(relaxed_choice(response,self.choices))

    def test_consistent_repetition_final_label_and_answer_text_sentence(self):
        for response in ['B: B','Final answer: B B.','Final: B','Library is the correct answer.']:
            self.assertEqual(relaxed_choice(response,self.choices),'B')
        choices=[dict(label=k,text=v) for k,v in zip('ABCDE',['shop','a library','bank','park','station'])]
        self.assertEqual(relaxed_choice('Answer: a library',choices),'B')
        self.assertIsNone(relaxed_choice('B: C',self.choices))


if __name__=='__main__':unittest.main()
