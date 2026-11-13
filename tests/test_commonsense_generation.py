"""Verify the spoken multiple-choice content and stable source identities."""

import json
import unittest

from cloud_tts_smoke import azure_ssml
from generate_commonsense_azure import questions,STYLES


class CommonsenseGenerationTests(unittest.TestCase):
    def test_same_question_and_every_choice_without_answer_key_in_all_styles(self):
        row=dict(id='source-id',answerKey='D',question=dict(stem='Where would someone read?',
            choices=[dict(label=k,text=v) for k,v in zip('ABCDE',['bank','shop','park','library','station'])]))
        selected=questions(json.dumps(row).encode(),1)[0]
        self.assertEqual(selected['answer'],'D')
        self.assertEqual(selected['answer_text'],'library')
        self.assertEqual(selected['question_id'],'commonsenseqa_validation_source-id')
        self.assertEqual(selected['source_index'],0)
        self.assertEqual(selected['question'],'Where would someone read?\nOption A: bank.\nOption B: shop.\nOption C: park.\nOption D: library.\nOption E: station.')
        for style in STYLES:
            ssml=azure_ssml(selected['question'],style,'en-US-JennyNeural',1.5)
            self.assertIn(selected['question'],ssml)
            self.assertNotIn('answerKey',ssml)
            self.assertNotIn('correct answer',ssml)
        self.assertIn('style="cheerful"',azure_ssml(selected['question'],'happy','en-US-JennyNeural',1.5))

    def test_rejects_unlabeled_data_and_wrong_choice_order(self):
        row=dict(id='x',answerKey='A',question=dict(stem='Question?',choices=[dict(label='B',text='b')]))
        with self.assertRaises(ValueError):questions(json.dumps(row).encode(),1)


if __name__=='__main__':unittest.main()
