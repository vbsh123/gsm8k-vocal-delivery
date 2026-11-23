import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import phi4_gsm_old_prompt as old
from evaluate_gsm_candidates import EXPLICIT,ORIGINAL


class OldPromptAblationTests(unittest.TestCase):
    def fixture(self,root):
        train=[dict(question_id=f'train-{i}') for i in range(425)]
        test=[dict(question_id=f'test-{i}',**{s:dict(question_id=f'test-{i}',voice='Jenny',
                  audio_sha256=f'{i}-{s}',reference_answer='2',audio_file=f'/fresh/{i}-{s}.wav')
                  for s in ('neutral','angry')}) for i in range(718)]
        parent=dict(task='gsm8k',collection=train,test=test,prompt=EXPLICIT,seed=42,max_new_tokens=512,
                    recipe=dict(model='phi',revision='pinned'),bank={p['question_id']:dict(answer='2') for p in test})
        source=root/'source.json';source.write_text(json.dumps(parent))
        return source,parent

    def test_old_prompt_only_keeps_fresh_audio_ids_recipe_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,parent=self.fixture(root)
            plan=old.prepare(source,root/'out')
            self.assertEqual(plan['prompt'],ORIGINAL)
            self.assertEqual(plan['recipe'],parent['recipe'])
            self.assertEqual(plan['max_new_tokens'],parent['max_new_tokens'])
            self.assertEqual(len(plan['recordings']),1436)
            self.assertEqual(plan['test_ids'],[p['question_id'] for p in parent['test']])
            self.assertTrue(all(r['split']=='test' and r['question_id'] not in plan['excluded_collection_ids'] for r in plan['recordings']))
            self.assertEqual(plan['recordings'][0]['audio_sha256'],parent['test'][0]['neutral']['audio_sha256'])

    def test_no_collection_no_steering_and_resume_skips_saved_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,_=self.fixture(root)
            plan=old.prepare(source,root/'out')
            plan['recordings']=plan['recordings'][:4]
            path=root/'out/run_config.json';path.write_text(json.dumps(plan))
            def generate(native,row,prompt,**kwargs):
                self.assertEqual(prompt,ORIGINAL)
                self.assertEqual(kwargs,dict(max_new_tokens=512))
                return dict(response='#### 2',response_tokens=3)
            with patch('phi4_all_layer_diagnostic.generate',side_effect=generate) as call:
                old.run(plan,root/'out',None)
                old.run(plan,root/'out',None)
                self.assertEqual(call.call_count,4)
            self.assertEqual(len(old.answers(plan,root/'out')),4)
            result=root/'out/results.jsonl'
            rows=[json.loads(line) for line in result.read_text().splitlines()]
            rows[0]['prompt']=EXPLICIT
            result.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'settings changed'):
                old.answers(plan,root/'out')


if __name__=='__main__':unittest.main()
