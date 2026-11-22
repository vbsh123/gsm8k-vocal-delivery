import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import diffa_happy_nightly as happy
import diffa_original_sad as sad
from tests import test_diffa_happy_nightly as happy_tests


class OriginalSadTests(unittest.TestCase):
    def fixture(self,root):
        out,_,inventory,target = happy_tests.HappyNightlyTests().fixture(root)
        with patch.object(happy.winners,'load_dataset',return_value=target):
            template = happy.task_plan('ifeval',dict(run=target['run']),out,inventory)
        for q in ('train','test'):
            path = root/'audio'/f'{q}-sad.wav'
            with wave.open(str(path),'wb') as audio:
                audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(16000)
                audio.writeframes(b'\1\0'*16)
            inventory[q,'sad'] = [(dict(question_id=q,voice=happy.VOICE,provider='azure',style='sad',
                                        question='question '+q,answer='2',audio_path=path.name),path.parent)]
        return template,out/'ifeval/run_config.json',inventory

    def test_sad_preserves_exact_cohort_and_recipe_without_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            template,source,inventory=self.fixture(root)
            output=root/'sad'
            plan=sad.prepare('ifeval',source,output,inventory)
            for key in ('collection_ids','test_ids','prompt','collection_prompt','max_new_tokens','bank'):
                self.assertEqual(plan[key],template[key])
            self.assertEqual(plan['pairs'][0]['neutral'],template['pairs'][0]['neutral'])
            self.assertEqual(plan['pairs'][0]['emotion']['style'],'sad')
            self.assertEqual({r['condition'] for r in plan['recordings']},{'neutral','sad'})
            self.assertEqual(plan,sad.prepare('ifeval',source,output,inventory))
            # No classification file exists or is required by collection selection.
            self.assertFalse(list(root.rglob('results-emotion2vec.jsonl')))
            with patch.object(happy,'generate',return_value=('#### 2',dict(mode='native'))) as generate:
                delivery=output/'delivery';delivery.mkdir()
                tokenizer=SimpleNamespace(encode=lambda *a,**k:[1,2])
                happy.deliver(plan,delivery,None,tokenizer,None)
                happy.deliver(plan,delivery,None,tokenizer,None)
                self.assertEqual(generate.call_count,1)
                self.assertEqual(generate.call_args.args[0]['style'],'sad')
            with patch.object(happy.winners,'digest',return_value='vectorhash'),patch.object(
                    happy.winners,'read_vector',return_value=(None,{})) as vector:
                job=sad.steering_job(plan,output)
                self.assertEqual(vector.call_args.args[1],'sad')
                self.assertEqual(job['strengths'],[0.,1.])
                self.assertEqual((job['layer'],job['group']),(31,'answer'))
            # Report uses sad's answers, with the same paired neutral baseline.
            rows=[dict(question_id='test',voice=happy.VOICE,condition=condition,response='answer',response_tokens=2,
                       metrics={'flexible-numeric':correct}) for condition,correct in [('neutral',False),('sad',True)]]
            with patch.object(happy.winners,'score_rows',return_value=(rows,{})):
                happy.report_delivery(dict(plan,name='gsm8k'),delivery,tokenizer,rows=rows)
            summary=json.loads((delivery/'summary.json').read_text())
            self.assertEqual(summary['paired'][0]['gain_pp'],100.)
            self.assertEqual(set(summary['lengths']),{'neutral','sad'})

    def test_missing_sad_never_changes_questions_or_generates_tts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _,source,inventory=self.fixture(root)
            inventory.pop(('train','sad'))
            with self.assertRaisesRegex(ValueError,'No TTS or question substitution'):
                sad.prepare('ifeval',source,root/'sad',inventory)


if __name__ == '__main__':
    unittest.main()
